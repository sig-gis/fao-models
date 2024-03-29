import numpy as np
import tensorflow as tf
import rasterio
import os

def convert_to_tf_dataset(input_data, labels):
    """
    Converts a pair of input array and labels array into a tf.Dataset of format tuple(inputs,targets).
    """

    # Convert numpy arrays to TensorFlow Datasets
    inputs_dataset = tf.data.Dataset.from_tensor_slices(input_data)
    targets_dataset = tf.data.Dataset.from_tensor_slices(labels)

    # Zip the inputs and targets together to get a dataset of (input, target) pairs,t
    dataset = tf.data.Dataset.zip((inputs_dataset, targets_dataset))
    return dataset

def one_hot_encode_binary(inputs, targets):
    """One-hot encode the target labels, must be binary targets."""
    targets_one_hot = tf.one_hot(tf.cast(targets,dtype=tf.uint8), depth=2)
    return inputs, targets_one_hot

def train_val_test_split_take_skip(dataset:tf.data.Dataset, size, train_split, val_split, test_split):
    # possible way to split tf.Dataset into train/val/test tf.Dataset's with shuffling
    # https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
    if int(sum([train_split, val_split, test_split])) != 1:
        raise ValueError('Train, validation, and test split must sum to 1.')
    
    buff = int(size/10) # don't full understand what logic goes into determining this. 
    full_dataset = dataset.shuffle(buffer_size=buff,reshuffle_each_iteration=False)
    
    train_size = int(train_split * size)
    val_size = int(val_split * size)
    test_size = int(test_split * size)
            
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    return train_dataset, val_dataset, test_dataset

def train_test_split_filter_map(dataset:tf.data.Dataset):
    # possible way to split tf.Dataset into train/test tf.Dataset's with shuffling
    # https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
    
    buff = 50 # don't full understand what logic goes into determining this. 
    full_dataset = dataset.shuffle(buffer_size=buff,reshuffle_each_iteration=False)
    
    # take every 4th element for test, rest for train, a 75/25 split
    def is_test(x, y):
        return x % 4 == 0

    def is_train(x, y):
        return not is_test(x, y)

    recover = lambda x,y: y

    test_dataset = full_dataset.enumerate() \
                        .filter(is_test) \
                        .map(recover)

    train_dataset = full_dataset.enumerate() \
                        .filter(is_train) \
                        .map(recover)
    return train_dataset, test_dataset

def get_geotiff_data_single_label(folder_path, num_geotiffs=100, batch_size=None):
    """Data loader; reads GeoTiff files from a folder and returns a TensorFlow Dataset."""
    image_patches = []
    labels = []

    for filename in os.listdir(folder_path)[:num_geotiffs]:
        if filename.endswith(".tif"):
            try:
                file_path = os.path.join(folder_path, filename)
                with rasterio.open(file_path) as dataset:
                    raster_data = dataset.read() / 10000  # Normalizing the raster data
                    if raster_data.shape != (4, 32, 32):
                        continue
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue

            raster_data = np.transpose(raster_data, (1, 2, 0))
            raster_data = np.expand_dims(raster_data, axis=0)

            label = 1 if 'nonforest' in filename else 0  # Binary label
            labels.append(label)
                    
            
            image_patches.append(raster_data)

    input_data = np.concatenate(image_patches, axis=0)
    labels = np.array(labels)

    return convert_to_tf_dataset(input_data, labels, batch_size)

def convert_to_tf_dataset(input_data, labels, batch_size):
    # Convert numpy arrays to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((input_data, labels))
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


## TESTING ################################################################
if __name__=="__main__":
    # Path to the folder containing GeoTiff files
    data_path = r"C:\fao-models\data"
    
    # Number of GeoTiffs to read
    num_tiffs = 100
    batch_size = 10
    
    all_data = get_geotiff_data_single_label(data_path,num_tiffs)
    all_data = all_data.map(one_hot_encode_binary)
    #print(all_data)
    #exit()

    # check the data out
    print('training data type',type(all_data))
    print(all_data)
    for element in all_data:
        print(element)
        break
    
    # train,test,val = train_val_test_split_take_skip(all_data,num_tiffs,0.7,0.15,0.15)
    # print('first splitting method')
    # print(train)
    # print(test)
    # print(val)
    # print('second splitting method')
    # train2,test2 = train_test_split_filter_map(all_data)
    # print(train2)
    # print(test2)
