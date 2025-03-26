import torch

from model.encoder import Prithvi_Encoder
from model.decoder import MLPMTCls

def build_change_detection_model(encoder_config,decoder_config,model_checkpoint):
    prithvi = Prithvi_Encoder(
        encoder_weights=encoder_config['encoder_weights'],
        download_url=encoder_config['download_url'],
        input_bands=encoder_config['input_bands'],
        input_size=encoder_config['input_size'],
        output_dim=encoder_config['output_dim'],
        patch_size=encoder_config['patch_size'],
        tubelet_size=encoder_config['tubelet_size'],
        in_chans=encoder_config['in_chans'],
        embed_dim=encoder_config['embed_dim'],
        output_layers=encoder_config['output_layers'],
        num_heads=encoder_config['num_heads'],
        depth=encoder_config['depth'],
        mlp_ratio=encoder_config['mlp_ratio'],
        num_frames=encoder_config['num_frames']
    )

    clf = MLPMTCls(
        encoder=prithvi,
        num_classes=decoder_config['num_classes'],
        topology=decoder_config['topology'],
        finetune=False,
        multi_temporal=decoder_config['multi_temporal'],
        multi_temporal_strategy=decoder_config['multi_temporal_strategy'],
        pooling_strategy=decoder_config['pooling_strategy'],
        softmax=decoder_config['softmax']
    )

    ckpt = torch.load(model_checkpoint,map_location=torch.device('cpu'))
    clf.load_state_dict(ckpt,strict=False)
    
    return clf