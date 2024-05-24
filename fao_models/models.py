import io
import itertools

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as backend
from tensorflow.keras.metrics import categorical_accuracy
import matplotlib.pyplot as plt
import keras
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names, count_forest, count_nonforest):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix: F-{count_forest} NF-{count_nonforest}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


class CmCallback(keras.callbacks.Callback):
    def __init__(self, test_labels, test_images, class_names, file_writer_cm):
        super().__init__()
        self.test_labels = test_labels
        self.test_images = test_images
        self.class_names = class_names
        self.file_writer_cm = file_writer_cm

    def on_epoch_end(self, epoch, logs=None):
        self.log_confusion_matrix(epoch=epoch)

    def log_confusion_matrix(self, epoch):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.test_images)
        test_pred = np.rint(test_pred_raw)
        count_nf = sum(self.test_labels.numpy() == 0)
        count_f = sum(self.test_labels.numpy() == 1)

        # Calculate the confusion matrix.
        cm = confusion_matrix(self.test_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(
            cm,
            class_names=self.class_names,
            count_forest=count_f,
            count_nonforest=count_nf,
        )
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("epoch_confusion_matrix", cm_image, step=epoch)


# metrics
def recall_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision


def f1_m(y_true, y_pred):
    __name__ = "f1_m"
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + backend.epsilon()))


def dice_coef(y_true, y_pred, smooth=1):
    __name__ = "dice_coef"
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred, smooth=1):
    __name__ = "dice_loss"
    y_true = tf.cast(y_true, tf.float32)
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=-1)
    true_sum = backend.sum(backend.square(y_true), -1)
    pred_sum = backend.sum(backend.square(y_pred), -1)
    return 1 - ((2.0 * intersection + smooth) / (true_sum + pred_sum + smooth))


evaluation_metrics = [categorical_accuracy, f1_m, precision_m, recall_m]


def resnet(training_mode=True):

    input_shape = (32, 32, 4)
    # Load the base ResNet model without the top (classifier) layers and without pre-trained weights
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,  # No pre-trained weights
        classes=2,
        input_shape=input_shape,  # Directly use 4-channel input shape
    )

    # Custom model
    inputs = layers.Input(shape=input_shape)

    # Pass the input to the base model
    x = base_model(
        inputs, training=training_mode
    )  # Set training=True to enable BatchNormalization layers

    # Add custom top layers
    # x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Use 'softmax' for multi-class
    # Create the final model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def mobilenet_v3small():
    input_shape = (32, 32, 4)

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        classes=2,
        include_preprocessing=False,
    )
    # Custom model
    inputs = layers.Input(shape=input_shape)

    # Pass the input to the base model
    x = base_model(
        inputs, training=True
    )  # Set training=True to enable BatchNormalization layers

    # Add custom top layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    # Create the final model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def vgg16():
    input_shape = (32, 32, 4)
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        classes=2,
    )

    # Custom model
    inputs = layers.Input(shape=input_shape)

    # Pass the input to the base model
    x = base_model(
        inputs, training=True
    )  # Set training=True to enable BatchNormalization layers

    # Add custom top layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    # Create the final model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model


bce_sum = tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction="sum",
)
# Create a dictionary of keyword-function pairs
model_dict = {
    "resnet": resnet,
    "mobilenet_v3small": mobilenet_v3small,
    "vgg16": vgg16,
}

custom_loss_funcs = {"dice_loss": dice_loss, "bce_sum": bce_sum}


def freeze(model):
    """Freeze model weights in every layer."""
    for layer in model.layers:
        layer.trainable = False

        if isinstance(layer, keras.models.Model):
            freeze(layer)
    model.trainable=False
    return model

def unfreeze(model):
    """Unfreeze model weights in every layer."""
    for layer in model.layers:
        layer.trainable = True

        if isinstance(layer, keras.models.Model):
            unfreeze(layer)
    model.trainable=True
    return model

def get_model(model_name, **kwargs):
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        dice_coef
    ]
    if model_name in model_dict:
        print(f"Model found: {model_name}")
        model_function = model_dict[model_name]
        loss = custom_loss_funcs.get(kwargs["loss_fn"], kwargs["loss_fn"])
        model = model_function(training_mode = kwargs["training_mode"])
        model.compile(optimizer=kwargs["optimizer"], loss=loss, metrics=metrics)
        return model
    else:
        raise NotImplementedError(f"Model '{model_name}' not found.")


if __name__ == "__main__":
    # Example usage
    model_name = "resnet"
    optimizer = "adam"
    loss_fn = "binary_crossentropy"
    metrics = ["accuracy"]
    m = get_model(model_name, optimizer=optimizer, loss_fn=loss_fn, metrics=metrics, training_mode=False)
    m.summary()
