import numpy as np
import datetime
import logging
from models import get_model, freeze
import os
import tensorflow as tf
import yaml
import argparse
import dataloader as dl


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
    model = get_model(model_name, optimizer=optimizer, loss_fn=loss_function, training_mode=False)
    model.load_weights(weights)
    freeze(model) # freeze model layers before loading weights

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
    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    seed = config["seed"]
    total_examples = config["total_examples"]
    test_split = config["test_split"]
    val_split = config["val_split"]
    model = load_predict_model(model_name, optimizer, loss_function, weights)

    # Load the dataset without batching
    dataset = dl.load_dataset_from_tfrecords(data_dir, batch_size=batch_size, buffer_size=buffer_size, seed=seed)

    # Split the dataset 2 ways or 3 ways
    if val_split is not None:
        train_dataset, test_dataset, val_dataset = dl.split_dataset(
            dataset,
            total_examples,
            test_split=test_split,
            batch_size=batch_size,
            val_split=val_split,
        )
        
    else:
        train_dataset, test_dataset = dl.split_dataset(
        dataset, total_examples, test_split=test_split, batch_size=batch_size
        )
   
    model.evaluate(val_dataset)

    y_pred = model.predict(val_dataset).flatten()
    print(len(y_pred))
    y_true = np.array([y for x, y in val_dataset.unbatch()])
    print(len(y_true))
    
    print(list(zip(y_true,y_pred))[0:50])




# main("dev-predict-runc-resnet-jjd.yml")
cli()
