import os
import math
from enum import Enum, auto

import numpy as np
import rasterio
import tensorflow as tf
from tqdm import tqdm


class SplitStrategy(Enum):
    all = auto()
    balanced = auto()


def balance_files(files: list[str]) -> list[str]:
    forest = []
    nonforest = []
    for i in files:
        if "non" in i:
            nonforest.append(i)
        else:
            forest.append(i)
    min_recs = min(len(forest), len(nonforest))
    return nonforest[:min_recs] + forest[:min_recs]


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, label):
    feature = {
        "image": _bytes_feature(image),
        "label": _bytes_feature(tf.io.serialize_tensor(label)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_sharded_tfrecords(
    folder_path,
    output_dir,
    items_per_record=500,
    balance_strat: SplitStrategy = SplitStrategy.all,
):
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
    if balance_strat.name == "balanced":
        filenames = balance_files(filenames)

    num_geotiffs = len(filenames)
    num_shards = math.ceil(num_geotiffs / items_per_record)

    options = tf.io.TFRecordOptions(compression_type="GZIP")

    for shard_id in range(num_shards):
        shard_filename = os.path.join(
            output_dir, f"geotiffs_shard_{shard_id:03d}.tfrecord.gz"
        )

        start_index = shard_id * items_per_record
        end_index = min((shard_id + 1) * items_per_record, num_geotiffs)

        with tf.io.TFRecordWriter(shard_filename, options=options) as writer:
            for filename in tqdm(filenames[start_index:end_index]):
                file_path = os.path.join(folder_path, filename)
                try:
                    with rasterio.open(file_path) as dataset:
                        raster_data = dataset.read() / 10000
                        if raster_data.shape != (4, 32, 32):
                            continue
                        raster_data = np.transpose(raster_data, (1, 2, 0))
                        image_raw = tf.io.serialize_tensor(
                            raster_data.astype(np.float32)
                        )
                        label = 1 if "nonforest" in filename else 0
                        # print(filename)
                        # print(label)
                        label = tf.convert_to_tensor(label, dtype=tf.int64)
                        tf_example = serialize_example(image_raw, label)
                        writer.write(tf_example)
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue

        # Print a message after finishing writing each TFRecord file
        print(f"Finished writing {shard_filename}")


if __name__ == "__main__":
    tiff_path = r"data"
    tf_path_root = r"tfrecords"
    balance_strat = SplitStrategy.all
    items_per_record = 1000  # Number of GeoTIFFs to store in each TFRecord file

    tf_path_strat = os.path.join(tf_path_root, balance_strat.name)
    for folder in [tiff_path, tf_path_root, tf_path_strat]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    write_sharded_tfrecords(
        tiff_path,
        tf_path_strat,
        items_per_record=items_per_record,
        balance_strat=balance_strat,
    )
