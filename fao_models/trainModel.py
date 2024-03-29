import tensorflow as tf
import os

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




# Example model (replace this with your actual model)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(32, 32, 4)),  # Adjust input shape as needed
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Define the path to your TFRecord files and set the batch size
# tfrecord_dir = "/home/ate/sig/gitmodels/fao-models/tfrecords"
tfrecord_dir = r"C:\fao-models\tfrecords"

batch_size = 32 # 256
split = 0.2
epochs = 1000 # 100
# Load the dataset without batching
dataset = load_dataset_from_tfrecords(tfrecord_dir)

# Split the dataset into training and testing
total_examples = 18067  # Update this with the actual number of examples in your dataset
train_dataset, test_dataset = split_dataset(dataset, total_examples, test_split=split, batch_size=batch_size)

LOGS_DIR = '.\logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
# Now you can use train_dataset for training and test_dataset for testing
steps_per_epoch = total_examples*split // batch_size
print(steps_per_epoch)

# The code above sets a tf.keras.optimizers.schedules.InverseTimeDecay to 
# hyperbolically decrease the learning rate to 1/2 of the base rate at 1,000 epochs, 1/3 at 2,000 epochs, and so on.
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.01,
  decay_steps=steps_per_epoch*epochs,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[tf.keras.callbacks.TensorBoard(LOGS_DIR)]
)




