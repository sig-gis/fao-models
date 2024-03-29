import tensorflow as tf
from tensorflow.keras import layers, models

def _parse_function(proto):
    # Define the parsing schema
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input `tf.train.Example` proto using the schema
    example = tf.io.parse_single_example(proto, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    label = tf.io.parse_tensor(example['label'], out_type=tf.int64)
    image.set_shape([32, 32, 4])  # Set the shape explicitly if not already defined
    label.set_shape([])  # For scalar labels
    return image, label

def load_dataset_from_tfrecords(tfrecord_dir, batch_size=32):

    pattern = tfrecord_dir + "/*.tfrecord.gz"
    files = tf.data.Dataset.list_files(pattern)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=tf.data.AUTOTUNE,
        block_length=1
    )
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    return dataset

def split_dataset(dataset, total_examples, test_split=0.2, batch_size=32):
    test_size = int(total_examples * test_split)
    train_size = total_examples - test_size

    # Calculate the number of batches for train and test sets
    train_batches = train_size // batch_size
    test_batches = test_size // batch_size

    train_dataset = dataset.take(train_batches).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_batches).take(test_batches).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset




def create_resnet_with_4_channels(input_shape=(32, 32, 4), num_classes=1):
    # Load the base ResNet model without the top (classifier) layers and without pre-trained weights
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,  # No pre-trained weights
        input_shape=input_shape  # Directly use 4-channel input shape
    )

    # Custom model
    inputs = layers.Input(shape=input_shape)

    # Pass the input to the base model
    x = base_model(inputs, training=True)  # Set training=True to enable BatchNormalization layers

    # Add custom top layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)  # Use 'softmax' for multi-class

    # Create the final model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Change to 'categorical_crossentropy' for multi-class
        metrics=['accuracy']
    )

    return model

# Example usage:
model = create_resnet_with_4_channels(input_shape=(32, 32, 4), num_classes=1)
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Define the path to your TFRecord files and set the batch size
tfrecord_dir = "/home/ate/sig/gitmodels/fao-models/tfrecords"
tfrecord_dir = r"C:\fao-models\tfrecords"
batch_size = 32

# Load the dataset without batching
dataset = load_dataset_from_tfrecords(tfrecord_dir)

# Split the dataset into training and testing
total_examples = 18067  # Update this with the actual number of examples in your dataset
train_dataset, test_dataset = split_dataset(dataset, total_examples, test_split=0.2, batch_size=batch_size)

# Now you can use train_dataset for training and test_dataset for testing
model.fit(train_dataset, epochs=200, validation_data=test_dataset)
