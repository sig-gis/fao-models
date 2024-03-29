#%%
# this was Ates slight adjustment to my model_fit which was using dataloader_from_gtiff methods 
# we now use using Ates data export->tfrecord->load tf.Dataset workflow now which is in 
from fao_models.graveyard.dataloader_from_gtiff import *
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers
from tensorflow.data import Dataset
from tensorflow.keras import backend as backend
from tensorflow.keras.metrics import categorical_accuracy
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

    # NO DECODER BLOCK
    
    # with this i was getting error: ValueError: `logits` and `labels` must have the same shape, received ((None, 1, 1, 2) vs (None, 32, 32)).
    # outputs = layers.Conv2D(2, (1, 1), activation='softmax')(center) # 2 values output
    
    # with this, not one-hot encoded labels, and binary_crossentropy loss fn, the model will train but not correctly.
    # outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(center) # 1 value output

    # to try and solve ValueError: Shapes (None, 32, 32, 2) and (None, 1, 1, 2) are incompatible, 
    # do we need to flatten the center layer with Dense?
    outputs = layers.Dense(2, activation='softmax')(center) # 2 values output
    # now getting ValueError: Shapes (None, 1, 2) and (None, 1, 1, 2) are incompatible
    outputs = tf.squeeze(outputs) # Kyle Dormans suggestion
    model = models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # Ate used dice_loss, found binary_crossentropy for binary classification in a tutorial..
        metrics=evaluation_metrics
    )

    return model

model = get_model()
# print(model.summary())
#%%
# Model training
# Path to the folder containing GeoTiff files
# training_path = r"C:\fao-models\data\training"
# validation_path = r"C:\fao-models\data\validation"
data_path = r"/home/ate/sig/gitmodels/fao-models/data/"

# Number of GeoTiffs to read
num_tiffs = 7500
batch_size = 64
epochs = 76

all_data = get_geotiff_data_single_label(data_path,num_tiffs)
all_data = all_data.map(one_hot_encode_binary)

# Split the data into training and validation sets
train,val = train_test_split_filter_map(all_data)
train = train.batch(batch_size)
val = val.batch(batch_size)

# training_data = get_geotiff_data_single_label(training_path,num_tiffs,batch_size)
# training_data = training_data.map(one_hot_encode_binary)

# validation_data = get_geotiff_data_single_label(validation_path,num_tiffs,batch_size)
# validation_data = validation_data.map(one_hot_encode_binary)

# check the data out
# print('training data type',type(training_data))
# print(training_data)
# for element in training_data:
#     print(element)
#     break
# %%
LOGS_DIR = '.\logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)]

model.fit(
train,
validation_data=val,
epochs=epochs,
shuffle=True,
callbacks=[tf.keras.callbacks.TensorBoard(LOGS_DIR)]
)
# %%
# saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
# if not os.path.exists(saved_models_dir):
#     os.makedirs(saved_models_dir)
# model.save(os.path.join(saved_models_dir,'modelv1_10k_train_test_set'))
