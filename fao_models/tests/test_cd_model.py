#!/usr/bin/env python

"""Tests for `fao_models` package."""

import pytest
from pathlib import Path

@pytest.fixture
def config():
    return Path("fao_models/configs/inference23_pytest.yml")

@pytest.fixture
def model(config):
    import yaml
    from fao_models.fao_models.model.util import build_change_detection_model
    from pprint import pprint
    with open(config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    with open(config['cd']['encoder'], 'r') as encoder_file:
        encoder_config = yaml.safe_load(encoder_file)
    with open(config['cd']['decoder'], 'r') as decoder_file:
        decoder_config = yaml.safe_load(decoder_file)
    pprint(config)
    pprint(encoder_config)
    pprint(decoder_config)
    weights = config['cd']['weights']

    # load model
    model = build_change_detection_model(encoder_config,decoder_config,weights)
    return  model

@pytest.fixture
def mean_std():
    import numpy as np
    means = np.array([0.05278337,0.08498019,0.10346901,0.2802707,0.25964622 ,0.16640756])
    stds = np.array([0.03278688, 0.05424733 ,0.08996119 ,0.07969411 ,0.12222017 ,0.12167657])
    return means,stds

@pytest.fixture
def classes():
    classes ={
    'Stable Non Forest': 0,
    'Stable Forest': 1,
    'Forest Loss': 2,
    'Forest Gain': 3,
}
    return classes

def test_forward(model,mean_std):
    import numpy as np
    import torch
    means, stds = mean_std

    arr1 = np.ones((6,32,32),dtype=float)
    arr1_normed = (arr1 - means[:,None,None]) / stds[:,None,None]

    arr2 = np.zeros((6,32,32),dtype=float)
    arr2_normed = (arr2 - means[:,None,None]) / stds[:,None,None]

    arr1_normed = arr1_normed.astype(np.float32)
    arr2_normed = arr2_normed.astype(np.float32)

    input = torch.from_numpy(np.stack([arr1_normed,arr2_normed]))
    input = input.unsqueeze(0)
    input = torch.permute(input,(0,2,1,3,4))
    
    pred = model({'optical':input})
   
    assert pred.shape == torch.Size([1,4]), f"incorrect output shape {pred.shape}"

  






