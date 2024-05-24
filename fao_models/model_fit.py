# %%
import datetime
import logging
from models import get_model, freeze, CmCallback
import dataloader as dl
import os
import tensorflow as tf
import numpy as np

import yaml
from pprint import pformat
import argparse


# setup logging
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


def main():

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

    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # retrieve parameters
    experiment_name = config_data["experiment_name"]
    model_name = config_data["model_name"]
    total_examples = config_data["total_examples"]
    data_dir = config_data["data_dir"]
    test_split = config_data["test_split"]
    val_split = config_data["val_split"]
    seed = config_data["seed"]
    epochs = config_data["epochs"]
    learning_rate = config_data["learning_rate"]
    decay_rate = config_data["decay_rate"]
    batch_size = config_data["batch_size"]
    buffer_size = config_data["buffer_size"]
    optimizer = config_data["optimizer"]
    optimizer_use_lr_schedular = config_data["optimizer_use_lr_schedular"]
    loss_function = config_data["loss_function"]
    early_stopping_patience = config_data["early_stopping_patience"]
    checkpoint = config_data["checkpoint"]
    if args.test:
        total_examples = batch_size * 5
        experiment_name = f"TEST{experiment_name}"

    # hyperbolically decrease the learning rate to 1/2 of the base rate at 1,000 epochs, 1/3 at 2,000 epochs, and so on.
    if optimizer == "adam":
        if optimizer_use_lr_schedular:
            steps_per_epoch = total_examples * test_split // batch_size
            lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=learning_rate,
                decay_steps=steps_per_epoch * epochs,
                decay_rate=decay_rate,
                staircase=False,
            )
            logger.info(
                f"Using a learning rate schedule of InverseTimeDecay, decay_steps={steps_per_epoch*epochs}"
            )
            optimizer = tf.keras.optimizers.Adam(lr_schedule)
        else:
            logger.info(f"Using a constant learning rate of {learning_rate}")
            optimizer = tf.keras.optimizers.Adam(learning_rate)

    # pull model from config
    model = get_model(model_name, optimizer=optimizer, loss_fn=loss_function)
    print(model.summary())

    logger.info("Config file: %s", config_file)
    logger.info("Parameters:")
    logger.info(pformat(config_data))

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
        
    # checking data splits for class balance
    logger.info('Reporting class balance for each data split...')

    logger.info('All Data')
    y_true = np.concatenate([y for x, y in dataset], axis=0)
    logger.info('y_true count: %s',len(y_true))
    vals, counts = np.unique(y_true, return_counts=True)
    logger.info('vals, counts: %s',[vals, counts])
    
    logger.info('Train Data')
    y_true_train = np.concatenate([y for x, y in train_dataset], axis=0)
    logger.info('y_true count: %s',len(y_true_train))
    vals, counts = np.unique(y_true_train, return_counts=True)
    logger.info('vals, counts: %s',[vals, counts])

    logger.info('Test Data')
    y_true_test = np.concatenate([y for x, y in test_dataset], axis=0)
    logger.info('y_true count: %s',len(y_true_test))
    vals, counts = np.unique(y_true_test, return_counts=True)
    logger.info('vals, counts: %s',[vals, counts])

    logger.info('Val Data')
    y_true_val = np.concatenate([y for x, y in val_dataset], axis=0)
    logger.info('y_true count: %s',len(y_true_val))
    vals, counts = np.unique(y_true_val, return_counts=True)
    logger.info('vals, counts: %s',[vals, counts])
    
    train_dataset = train_dataset.shuffle(
        buffer_size, reshuffle_each_iteration=True)
    
    logger.info("Starting model training...")
    LOGS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "logs", experiment_name
    )
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    SAVED_MODELS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "saved_models", experiment_name
    )
    if not os.path.exists(SAVED_MODELS_DIR):
        os.makedirs(SAVED_MODELS_DIR)
    
    # setup for confusion matrix callback
    tb_samples = train_dataset.take(1)
    x = list(map(lambda x: x[0], tb_samples))[0]
    y = list(map(lambda x: x[1], tb_samples))[0]
    class_names = ["nonforest", "forest"]

    # initialize and add tb callbacks
    file_writer = tf.summary.create_file_writer(LOGS_DIR)
    tb_callback = tf.keras.callbacks.TensorBoard(LOGS_DIR)
    cm_callback = CmCallback(y, x, class_names, file_writer)
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
    )
    callbacks = [cm_callback, 
                 save_model_callback,
                 tb_callback]

    if early_stopping_patience is not None:
        logger.info(f"Using early stopping. Patience: {early_stopping_patience}")
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )
        callbacks.append(early_stop)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
    )

    logger.info("Model training complete")
    logger.info("Training history:")
    logger.info(pformat(history.history))
    
    # perform evaluation on validation set if it exists
    if val_split is not None:
        logger.info(f"loading model weights from checkpoint: {checkpoint}")
        model = get_model(model_name, optimizer=optimizer, loss_fn=loss_function)
        model.load_weights(checkpoint)
        freeze(model) # freeze weights for inference
        eval = model.evaluate(val_dataset,return_dict=True)
        logger.info(f"Validation: {pformat(eval)}")

if __name__ == "__main__":
    main()
