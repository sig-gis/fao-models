#!/usr/bin/env python

"""Tests for `fao_models` package."""

import pytest
import fao_models
from pathlib import Path

@pytest.fixture
def config():
    return Path("fao_models/configs/inference23_pytest.yml")

@pytest.fixture
def model(config):
    from fao_models.common import load_yml
    from fao_models.models import load_predict_model
    _config = load_yml(config)
    model = load_predict_model(
            model_name= _config["fnf"]["model_name"],
            optimizer= _config["fnf"]["optimizer"],
            loss_fn= _config["fnf"]["loss_function"],
            weights= _config["fnf"]["weights"],
        )
    return model

def test_forward(model):
    import numpy as np
    dummy_img = np.ones((1,32,32,4),dtype=float)
    forward = model(dummy_img)
    print(dummy_img)
    print(forward)
    assert forward.shape == (1,1), f"incorrect out shape {forward.shape}"






