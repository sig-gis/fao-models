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