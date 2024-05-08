import os
import tensorflow as tf
import dataloader as dl
from models import get_model
import yaml
import argparse
import numpy as np
from pprint import pformat


def main():

    # initalize new cli parser
    parser = argparse.ArgumentParser(description="Evaluate a model with a .yml file.")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to .yml file",
    )
    
    args = parser.parse_args()

    config_file = args.config

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # retrieve parameters
    model_name = config["model_name"]
    checkpoint = config["checkpoint"]
    optimizer = config["optimizer"]
    loss_function = config["loss_function"]

    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    val_split = config["val_split"]
    test_split = config["test_split"]
    total_examples = config["total_examples"]
    buffer_size = config["buffer_size"]

    # load model from checkpoint
    model = get_model(model_name, optimizer=optimizer, loss_fn=loss_function)
    model.load_weights(checkpoint)
    # when mode.trainable was set to False, was getting error at model.load_weights(checkpoint): ValueError: axes don't match array
    # https://stackoverflow.com/questions/51944836/keras-load-model-valueerror-axes-dont-match-array
    model.trainable = True 

    # print(model.summary())

    # load data 
    dataset = dl.load_dataset_from_tfrecords(data_dir, batch_size=batch_size, buffer_size=buffer_size)
    
    train_dataset, test_dataset, val_dataset = dl.split_dataset(
            dataset,
            total_examples,
            test_split=test_split,
            batch_size=batch_size,
            val_split=val_split,
    )
        

    model.load_weights(checkpoint)
    eval = model.evaluate(val_dataset,return_dict=True)
    print(f"Validation: {pformat(eval)}")



if __name__ == "__main__":
    main()
