#%%
from array_split_input_targets import get_geotiff_data
import os
import rasterio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers
from tensorflow.data import Dataset
from tensorflow.keras import backend as backend
from tensorflow.keras.metrics import categorical_accuracy

# data loader for training,validation,test data, 
def get_geotiff_data(folder_path, num_geotiffs=10):
    """Data loader; reads GeoTiff files from a folder and returns a TensorFlow Dataset."""
    # List to store the formatted training data batches
    training_data_batches = []

    # Iterate over the GeoTiffs in the folder
    for filename in os.listdir(folder_path)[:num_geotiffs]:
        if filename.endswith(".tif"):
            # Open the GeoTiff file, move on if can't open the file or the raster dataset doesn't have expected shape
            try:
                file_path = os.path.join(folder_path, filename)
                with rasterio.open(file_path) as dataset:
                    raster_data = dataset.read() # R G B N label # 
                    if raster_data.shape != (5, 32, 32): # Skip the GeoTiff if it does not have the expected shape
                        continue
                # print(raster_data.shape) # shape is (bands, rows, columns)
            except:
                continue
            
            # Reshape the raster data to match TensorFlow's input shape (batch_size, height, width, channels)
            raster_data = np.transpose(raster_data, (1, 2, 0))
            raster_data = np.expand_dims(raster_data, axis=0)
            
            # is this recommended? not doing it at moment
            # Normalize the raster data between 0 and 1
            # scaler = MinMaxScaler()
            # raster_data = scaler.fit_transform(raster_data.reshape(-1, raster_data.shape[-1]))
            
            raster_data = raster_data.reshape(raster_data.shape[0], raster_data.shape[1], raster_data.shape[2], -1)
            
            # Add the formatted training data batch to the list
            training_data_batches.append(raster_data)


    # Concatenate the training data batches along the batch axis
    training_data = np.concatenate(training_data_batches, axis=0)
    # print(training_data.shape)
    # print(training_data)

    # Convert numpy array to TensorFlow Dataset
    # training_data = tf.data.Dataset.from_tensor_slices(training_data) # - this was not a valid x for model.fit()
    
    # having trouble turning my numpy array of shape (batchsize, 32, 32, 5) into the tf.Dataset format of tuple(inputs, targets)
    # needed for model.fit() https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    
    # this might work? hard to tell what's causing the error in model.fit()
    def convert_to_tf_dataset(training_data):
        def split_inputs_targets(element):
            inputs = element[..., :-1]  # All but the last value in the innermost array
            targets = element[..., -1]  # The last value in the innermost array
            return inputs, targets

        inputs, targets = split_inputs_targets(training_data)

        # Convert numpy arrays to TensorFlow Datasets
        inputs_dataset = tf.data.Dataset.from_tensor_slices(inputs)
        targets_dataset = tf.data.Dataset.from_tensor_slices(targets)

        # Zip the inputs and targets together to get a dataset of (input, target) pairs
        dataset = tf.data.Dataset.zip((inputs_dataset, targets_dataset))
        return dataset
    
    return convert_to_tf_dataset(training_data)
#%%
# Model architecture
# Ate's crop mapping architecture - minus decoder blocks

optimizer = optimizers.Adam()

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
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + backend.epsilon()))

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1):
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=-1)
    true_sum = backend.sum(backend.square(y_true), -1)
    pred_sum = backend.sum(backend.square(y_pred), -1)
    return 1 - ((2. * intersection + smooth) / (true_sum + pred_sum + smooth))

evaluation_metrics = [categorical_accuracy, f1_m, precision_m, recall_m]

def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def get_model():
    inputs = layers.Input(shape=[None, None, 4])
    encoder0_pool, encoder0 = encoder_block(inputs, 16)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 64)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 128)
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 256)
    center = conv_block(encoder4_pool, 512)

    # no decoder block
    
    outputs = layers.Conv2D(2, (1, 1), activation='sigmoid')(center) # 2 values as output?
    # outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(center) # or 1 value output?

    model = models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy', # Ate used dice_loss, found binary_crossentropy for binary classification in a tutorial..
        metrics=evaluation_metrics
    )

    return model

model = get_model()
print(model.summary())
#%%
# Model training
# Path to the folder containing GeoTiff files
training_path = r"C:\fao-models\data\training"
validation_path = r"C:\fao-models\data\validation"

# Number of GeoTiffs to read
batch_size = 10
epochs = 10

training_data = get_geotiff_data(training_path,batch_size)
validation_data = get_geotiff_data(validation_path,batch_size)
print(training_data)

# %%
model.fit(
training_data,
validation_data=validation_data,
epochs=epochs,  # You might want to use the EPOCHS variable here
# callbacks=[tf.keras.callbacks.TensorBoard(LOGS_DIR)]
)
# %%
