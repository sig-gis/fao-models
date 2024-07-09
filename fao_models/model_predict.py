import numpy as np
import datetime
import logging
from models import get_model
import os
import tensorflow as tf
import rasterio as rio
import yaml
import argparse


logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.WARNING,
    filename=os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        f'trainlog_{datetime.datetime.now().strftime("%Y-%m-%d")}.log',
    ),  # add _%H-%M-%S if needbe
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cli():
    # initalize new cli parser
    parser = argparse.ArgumentParser(description="Train a model with a .yml file.")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to .yml file",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=bool,
        default=False,
        help="Run as a test. limits total examples to 5*batch_size and adds a test prefix to experiment name",
    )
    args = parser.parse_args()

    config_file = args.config
    main(config_file)


def load_predict_model(model_name, optimizer, loss_function, weights):
    model = get_model(model_name, optimizer=optimizer, loss_fn=loss_function)
    model.load_weights(weights)
    model.trainable = False
    return model


def main(config: str | dict):
    # load model
    if isinstance(config, str):
        with open(config, "r") as file:
            config = yaml.safe_load(file)
    model_name = config["model_name"]
    weights = config["checkpoint"]
    optimizer = config["optimizer"]
    loss_function = config["loss_function"]

    model = load_predict_model(model_name, optimizer, loss_function, weights)
    print(model.summary())
    # load image
    # local file as placeholder
    # img = "data/patch_pt9097_nonforest.tif"
    img = "data\\data_qa_old_caf\\patch_pt0_nonforest.tif"
    with rio.open(img) as dst:
        data = dst.read() / 10_000
        profile = dst.profile
    # convert to tensor
    data = np.transpose(data, (1, 2, 0))
    tfdata = tf.expand_dims(tf.convert_to_tensor(data.astype(np.float32)), axis=0)
    o = model.predict(tfdata)
    print(o)


# main("dev-predict-runc-resnet-jjd.yml")
cli()
