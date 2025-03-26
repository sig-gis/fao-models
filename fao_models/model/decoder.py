import torch.nn.functional as F
import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Sequence

from model.base import Decoder,Encoder
from model.ltae import LTAE2d, LTAEChannelAdaptor


from einops import rearrange

class MLP(Decoder):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: str,
        topology: Sequence[int],
        softmax: bool=True,
        interp_mode: str='bilinear',
        align_corners:bool=False,
        feature_multiplier: int = 1
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = 'MLP_Singletemporal_segmentation'
        self.encoder = encoder
        self.finetune = finetune
        self.feature_multiplier = feature_multiplier
        self.align_corners = align_corners
        self.interp_mode=interp_mode
        self.topology = topology
        self.softmax= softmax

        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.finetune == 'retrain_input':
            for param in self.encoder.parameters():
                param.requires_grad = False

            self.encoder.unfreeze_input_layer()

        if self.topology is None:
            self.topology = []
            self.topology.append(self.encoder.output_dim[-1])
            self.topology.append(self.num_classes)
        else:
            self.topology = [self.encoder.output_dim[-1]] + self.topology + [self.num_classes]

        self.n_layers = len(self.topology)


        self.layers = nn.ModuleList()
        for i in range(1,self.n_layers):
            if i == self.n_layers - 1:
                self.layers.extend(
                    [
                        nn.Linear(self.topology[i-1],self.topology[i]),
                        # nn.ReLU()
                    ]
                )
            else:
                self.layers.extend(
                    [
                        nn.Linear(self.topology[i-1],self.topology[i]),
                        nn.ReLU()
                    ]
                )
        self.mlp = nn.Sequential(*self.layers)
        # self.conv_seg = OutConv(self.topology[0], self.num_classes)

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T=1 H W) with C the number of encoders'bands for the given modality.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """


        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img)
            else:
                feat = self.encoder(img)

            if self.encoder.multi_temporal_output:

                feat = [f.squeeze(-3) for f in feat]
        else:
            # remove the temporal dim
            # [B C T=1 H W] -> [B C H W]
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
            else:
                feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})

        # feat = torch.stack(feat,dim=0)
        b, c, h, w = feat[-1].shape

        feat = rearrange(feat[-1],'b c h w -> (b h w) c')

        output = self.mlp(feat)
        
        if self.softmax:
            output = F.softmax(output,dim=1)
        # val, indices = torch.topk(output,dim=1,k=1)

        # output = indices

        output = rearrange(output,'(b h w) c -> b c h w',b=b,h=h,w=w)
        output = F.interpolate(output,size=output_shape,mode=self.interp_mode,align_corners=self.align_corners)

        return output

class MLPMT(Decoder):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        topology: Sequence[int],
        finetune: bool,
        multi_temporal: int,
        multi_temporal_strategy: str | None,
        softmax: bool,
        interp_mode: str='bilinear',
        align_corners:bool=False,
    ) -> None:
        decoder_in_channels = self.get_decoder_in_channels(
            multi_temporal_strategy, encoder
        )
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = 'MLP_multitemporal_segmentation'
        self.multi_temporal = multi_temporal
        self.multi_temporal_strategy = multi_temporal_strategy
        self.topology = topology
        self.softmax=softmax
        self.interp_mode=interp_mode
        self.align_corners = align_corners

        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.encoder.multi_temporal and not self.encoder.multi_temporal_output:
            self.tmap = None
        else:
            if self.multi_temporal_strategy == 'ltae':
                ltae_in_channels = decoder_in_channels

                self.ltae = LTAE2d(
                    positional_encoding=False,
                    in_channels=ltae_in_channels,
                    mlp=[ltae_in_channels,ltae_in_channels],
                    d_model=ltae_in_channels
                )

                if self.topology is None:
                    self.topology = []
                    self.topology.append(ltae_in_channels)
                    self.topology.append(self.num_classes)
                else:
                    self.topology = [ltae_in_channels] + self.topology + [self.num_classes]

                self.n_layers = len(self.topology)


                self.layers = nn.ModuleList()
                for i in range(1,self.n_layers):
                    self.layers.extend(
                        [
                            nn.Linear(self.topology[i-1],self.topology[i]),
                            # nn.ReLU()
                        ]
                    )
                self.mlp = nn.Sequential(*self.layers) 

            elif self.multi_temporal_strategy == 'linear':
                if self.topology is None:
                    self.topology = []
                    self.topology.append(ltae_in_channels*self.multi_temporal)
                    self.topology.append(self.num_classes)
                else:
                    self.topology = [ltae_in_channels*self.multi_temporal] + self.topology + [self.num_classes]

                self.n_layers = len(self.topology)

                self.layers = nn.ModuleList()
                for i in range(1,self.n_layers):
                    if i == self.n_layers - 1:
                        self.layers.extend(
                            [
                                nn.Linear(self.topology[i-1],self.topology[i]),
                                # nn.ReLU()
                            ]
                        )
                    else:
                        self.layers.extend(
                            [
                                nn.Linear(self.topology[i-1],self.topology[i]),
                                nn.ReLU()
                            ]
                        )
                self.mlp = nn.Sequential(*self.layers)
            else:
                if self.topology is None:
                    self.topology = []
                    self.topology.append(ltae_in_channels)
                    self.topology.append(self.num_classes)
                else:
                    self.topology = [ltae_in_channels] + self.topology + [self.num_classes]

                self.n_layers = len(self.topology)

                self.layers = nn.ModuleList()
                for i in range(1,self.n_layers):
                    if i == self.n_layers - 1:
                        self.layers.extend(
                            [
                                nn.Linear(self.topology[i-1],self.topology[i]),
                                # nn.ReLU()
                            ]
                        )
                    else:
                        self.layers.extend(
                            [
                                nn.Linear(self.topology[i-1],self.topology[i]),
                                nn.ReLU()
                            ]
                        )
                self.mlp = nn.Sequential(*self.layers)
                

    
    def get_decoder_in_channels(
        self, multi_temporal_strategy: str | None, encoder: Encoder
    ) -> int:
        if multi_temporal_strategy == "ltae" or multi_temporal_strategy =='linear':
            # if the encoder output channels vary we must use an adaptor before the LTAE
            decoder_in_channels = encoder.output_dim[-1]
        else:
            decoder_in_channels = encoder.output_dim[-1]
        return decoder_in_channels

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output for multi-temporal data.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T H W) with C the number of encoders'bands for the given modality,
            and T the number of time steps.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """
        # If the encoder handles multi_temporal we feed it with the input
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feats = self.encoder(img)
            else:
                feats = self.encoder(img)
            # multi_temporal models can return either (B C' T H' W')
            # or (B C' H' W') via internal merging strategy

        # If the encoder handles only single temporal data, we apply multi_temporal_strategy
        else:
            feats = []
            for i in range(self.multi_temporal):
                if not self.finetune:
                    with torch.no_grad():
                        feats.append(
                            self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                        )
                else:
                    feats.append(
                        self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                    )

            feats = [list(i) for i in zip(*feats)]
            # obtain features per layer
            feats = [torch.stack(feat_layers, dim=2) for feat_layers in feats]


        feats = feats[-1]
        
        b,c,t,h,w = feats.shape

        if self.multi_temporal_strategy == 'ltae':
            feats = self.ltae(feats)

            feats = rearrange(feats,'b c h w -> (b h w) c')
            output = self.mlp(feats)
        elif self.multi_temporal_strategy == 'linear':
            feats = rearrange(feats,'b c t h w -> (b h w) (c t)')
            output = self.mlp(feats)
        else:
            pass

        if self.softmax:
            output = F.softmax(output,dim=1)

        output = rearrange(output,'(b h w) c -> b c h w',b=b,h=h,w=w)
        output = F.interpolate(output,output_shape,mode=self.interp_mode,align_corners=self.align_corners)

        return output
    
class MLPMTCls(Decoder):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        topology: Sequence[int],
        finetune: bool,
        multi_temporal: int,
        multi_temporal_strategy: str | None,
        pooling_strategy:str | None,
        softmax: bool,
    ) -> None:
        decoder_in_channels = self.get_decoder_in_channels(
            multi_temporal_strategy, encoder
        )
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = 'MLP_multitemporal_classification'
        self.multi_temporal = multi_temporal
        self.multi_temporal_strategy = multi_temporal_strategy
        self.topology = topology
        self.softmax=softmax

        self.pooling_strategy = pooling_strategy

        print(self.encoder.multi_temporal_output)
        # self.num_patches = encoder.num_patches
        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.encoder.multi_temporal and not self.encoder.multi_temporal_output:
            ltae_in_channels = decoder_in_channels
            if self.topology is None:
                self.topology = []
                self.topology.append(ltae_in_channels)
                self.topology.append(self.num_classes)
            else:
                self.topology = [ltae_in_channels] + self.topology + [self.num_classes]

            self.n_layers = len(self.topology)


            self.layers = nn.ModuleList()
            for i in range(1,self.n_layers):
                self.layers.extend(
                    [
                        nn.Linear(self.topology[i-1],self.topology[i]),
                        # nn.ReLU()
                    ]
                )
            self.mlp = nn.Sequential(*self.layers)
        else:
            if self.multi_temporal_strategy == 'ltae':
                ltae_in_channels = decoder_in_channels

                self.ltae = LTAE2d(
                    positional_encoding=False,
                    in_channels=ltae_in_channels,
                    mlp=[ltae_in_channels,ltae_in_channels],
                    d_model=ltae_in_channels
                )

                if self.topology is None:
                    self.topology = []
                    self.topology.append(ltae_in_channels)
                    self.topology.append(self.num_classes)
                else:
                    self.topology = [ltae_in_channels] + self.topology + [self.num_classes]

                self.n_layers = len(self.topology)


                self.layers = nn.ModuleList()
                for i in range(1,self.n_layers):
                    self.layers.extend(
                        [
                            nn.Linear(self.topology[i-1],self.topology[i]),
                            # nn.ReLU()
                        ]
                    )
                self.mlp = nn.Sequential(*self.layers) 

            elif self.multi_temporal_strategy == 'linear':
                ltae_in_channels = decoder_in_channels
                if self.topology is None:
                    self.topology = []
                    self.topology.append(ltae_in_channels*self.multi_temporal)
                    self.topology.append(self.num_classes)
                else:
                    self.topology = [ltae_in_channels*self.multi_temporal] + self.topology + [self.num_classes]

                self.n_layers = len(self.topology)

                self.layers = nn.ModuleList()
                for i in range(1,self.n_layers):
                    if i == self.n_layers - 1:
                        self.layers.extend(
                            [
                                nn.Linear(self.topology[i-1],self.topology[i]),
                                # nn.ReLU()
                            ]
                        )
                    else:
                        self.layers.extend(
                            [
                                nn.Linear(self.topology[i-1],self.topology[i]),
                                nn.ReLU()
                            ]
                        )
                self.mlp = nn.Sequential(*self.layers)
            else:
                if self.topology is None:
                    self.topology = []
                    self.topology.append(ltae_in_channels)
                    self.topology.append(self.num_classes)
                else:
                    self.topology = [ltae_in_channels] + self.topology + [self.num_classes]

                self.n_layers = len(self.topology)

                self.layers = nn.ModuleList()
                for i in range(1,self.n_layers):
                    if i == self.n_layers - 1:
                        self.layers.extend(
                            [
                                nn.Linear(self.topology[i-1],self.topology[i]),
                                # nn.ReLU()
                            ]
                        )
                    else:
                        self.layers.extend(
                            [
                                nn.Linear(self.topology[i-1],self.topology[i]),
                                nn.ReLU()
                            ]
                        )
                self.mlp = nn.Sequential(*self.layers)
                

    
    def get_decoder_in_channels(
        self, multi_temporal_strategy: str | None, encoder: Encoder
    ) -> int:
        if multi_temporal_strategy == "ltae" or multi_temporal_strategy =='linear':
            # if the encoder output channels vary we must use an adaptor before the LTAE
            decoder_in_channels = encoder.output_dim[-1]
        else:
            decoder_in_channels = encoder.output_dim[-1]
        return decoder_in_channels

    def forward(
        self, img: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the segmentation output for multi-temporal data.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T H W) with C the number of encoders'bands for the given modality,
            and T the number of time steps.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """
        # If the encoder handles multi_temporal we feed it with the input
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feats = self.encoder(img)
            else:
                feats = self.encoder(img)
            # multi_temporal models can return either (B C' T H' W')
            # or (B C' H' W') via internal merging strategy

        # If the encoder handles only single temporal data, we apply multi_temporal_strategy
        else:
            feats = []
            for i in range(self.multi_temporal):
                if not self.finetune:
                    with torch.no_grad():
                        feats.append(
                            self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                        )
                else:
                    feats.append(
                        self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                    )

            feats = [list(i) for i in zip(*feats)]
            # obtain features per layer
            feats = [torch.stack(feat_layers, dim=2) for feat_layers in feats]


        feats = feats[-1]

        b,c,t,h,w = feats.shape

        if self.multi_temporal_strategy == 'ltae':
            feats = self.ltae(feats)
            # feats = rearrange(feats,'b c h w -> (b h w) c')
            if self.pooling_strategy == 'mean':
                feats = torch.mean(feats,dim=(-1,-2))
            elif self.pooling_strategy == 'max':
                feats = torch.amax(feats,dim=(-1,-2))
                
            output = self.mlp(feats)
        elif self.multi_temporal_strategy == 'linear':

            if self.pooling_strategy == 'mean':
                feats = torch.mean(feats,dim=(-1,-2))
            elif self.pooling_strategy == 'max':
                feats = torch.amax(feats,dim=(-1,-2))
            
            feats = rearrange(feats,'b c t -> b (c t)')
            output = self.mlp(feats)
        else:
            pass

        if self.softmax:
            output = F.softmax(output,dim=1)

        # output = rearrange(output,'(b h w) c -> b c h w',b=b,h=h,w=w)
        # output = F.interpolate(output,output_shape,mode=self.interp_mode,align_corners=self.align_corners)

        return output

