# Adapted from https://github.com/NASA-IMPACT/hls-foundation-os

from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Block

from model.base import Encoder

import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Prithvi_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2310.18660
    Attributes:
        output_layers (int | list[int]): The layers from which to extract the output.
        img_size (int): The size of the input image.
        num_frames (int): The number of frames in the input data.
        patch_size (int): The size of each patch.
        in_chans (int): The number of input channels.
        patch_embed (PatchEmbed): The patch embedding layer.
        cls_token (nn.Parameter): The class token parameter.
        pos_embed (nn.Parameter): The positional embedding parameter.
        blocks (nn.ModuleList): The list of Transformer blocks.
    Methods:
        __init__(self, encoder_weights: str | Path, input_bands: dict[str, list[str]], input_size: int, output_layers: int | list[int], patch_size=16, tubelet_size=1, in_chans=3, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, num_frames=1):
            Initializes the Prithvi_Encoder with the given parameters.
        load_encoder_weights(self, logger: Logger) -> None:
            Loads the encoder weights from a pretrained model and handles any missing or incompatible shapes.
        freeze(self):
            Freezes the parameters of the encoder to prevent them from being updated during training.
        initialize_weights(self):
            Initializes the weights of the encoder, including the positional embeddings and patch embeddings.
        _init_weights(self, m):
            Initializes the weights of the given module using Xavier uniform initialization for Linear layers and constant initialization for LayerNorm layers.
        forward(self, image):
            Performs the forward pass of the encoder, embedding the input patches, adding positional embeddings, and applying the Transformer blocks.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        output_dim: int | list[int],
        output_layers: int | list[int],
        download_url: str,
        patch_size=16,
        tubelet_size=1,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        num_frames=1,
        resize_pos_embed= False,
        ft_bands=None
    ):
        super().__init__(
            model_name="Prithvi",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=True,
            multi_temporal_output=True,
            pyramid_output=False,
            download_url=download_url,
        )

        self.output_layers = output_layers

        self.img_size = self.input_size
        self.tublet_size = tubelet_size

        if num_frames:
            self.num_frames = num_frames
        else:
            self.num_frames = 1

        self.patch_size = patch_size
        self.in_chans = in_chans

        self.resize_pos_embed = resize_pos_embed if resize_pos_embed != self.img_size else False
        if ft_bands is not None:
            self.ft_bands = None
            if len(ft_bands['optical']) != in_chans:
                self.ft_bands = len(ft_bands['optical'])
        else:
            self.ft_bands = None

        self.patch_embed = PatchEmbed(
            self.img_size,
            patch_size,
            self.num_frames,
            tubelet_size,
            in_chans,
            embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.initialize_weights()

        if self.resize_pos_embed or self.ft_bands:
            self.resize_input(self.resize_pos_embed,self.ft_bands)
            self.img_size = self.resize_pos_embed


    def resize_input(self,ft_img_size,ft_bands=None)->None:
        if self.ft_bands is not None:
            self.img_size = ft_img_size
            self.in_chans = ft_bands
            self.patch_embed = PatchEmbed(
                self.img_size,
                self.patch_size,
                self.num_frames,
                self.tublet_size,
                self.in_chans,
                self.embed_dim,
            )
            num_patches = self.patch_embed.num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim), requires_grad=False
            )

            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            ft_img_size = (ft_img_size,ft_img_size)
            self.patch_embed.grid_size = (
                self.patch_embed.num_frames // self.patch_embed.tubelet_size,
                ft_img_size[0] // self.patch_embed.patch_size[0],
                ft_img_size[1] // self.patch_embed.patch_size[1],
            )
            self.patch_embed.num_patches = self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2]

            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim), requires_grad=False
            )

            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        return self.pos_embed
    def unfreeze_input_layer(self):
        for param in self.patch_embed.parameters():
            param.requires_grad = True
    
    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)

    def load_encoder_weights_no_log(self) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        self.load_state_dict(pretrained_encoder, strict=False)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image):
        # embed patches
        x = image["optical"]
        x = self.patch_embed(x)

        

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        # apply Transformer blocks

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                out = (
                    x[:, 1:, :]
                    .permute(0, 2, 1)
                    .view(
                        x.shape[0],
                        -1,
                        self.num_frames,
                        self.img_size // self.patch_size,
                        self.img_size // self.patch_size,
                    )
                    .squeeze(2)
                    .contiguous()
                )
                output.append(out)

        return output


    def enforce_single_temporal(self):

        self.num_frames = 1

        self.patch_embed = PatchEmbed(
            self.input_size,
            self.patch_size,
            1,
            self.tublet_size,
            self.in_chans,
            self.embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False
        )

        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

class PatchEmbed(nn.Module):
    """Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x
