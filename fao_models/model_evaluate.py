import os
import tensorflow as tf
import dataloader as dl
from models import get_model
import yaml
import argparse

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

    val_data_dir = config["val_data_dir"]
    batch_size = config["batch_size"]
    

    # load model from checkpoint
    model = get_model(model_name, optimizer=optimizer, loss_fn=loss_function)
    model.load_weights(checkpoint)
    model.trainable = False

    print(model.summary())

    # load data 
    dataset = dl.load_dataset_from_tfrecords(val_data_dir, batch_size=batch_size)
    model.evaluate(dataset)

if __name__ == "__main__":
    main()
