import os
import rasterio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Path to the folder containing GeoTiff files
folder_path = r"C:\fao-models\data"

# Number of GeoTiffs to read
num_geotiffs = 800
epochs = 10

# List to store the formatted training data batches
training_data_batches = []

# Iterate over the GeoTiffs in the folder
for filename in os.listdir(folder_path)[:num_geotiffs]:
    if filename.endswith(".tif"):
        # Open the GeoTiff file
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
        # print('np.transposed',raster_data[0])
        raster_data = np.expand_dims(raster_data, axis=0)
        # print('np.expand_dims',raster_data[0][0])
        
        # Normalize the raster data between 0 and 1
        # scaler = MinMaxScaler()
        # raster_data = scaler.fit_transform(raster_data.reshape(-1, raster_data.shape[-1]))
        # print('fit_transform',raster_data[0])
        raster_data = raster_data.reshape(raster_data.shape[0], raster_data.shape[1], raster_data.shape[2], -1)

        # # Add the formatted training data batch to the list
        training_data_batches.append(raster_data)
    #break
# print('training_data_batches',training_data_batches)

# Concatenate the training data batches along the batch axis
training_data = np.concatenate(training_data_batches, axis=0)
# print('np.concatenate(trainin_data_batches)',training_data)
# Generate dummy labels for the training data
labels = np.random.randint(0, 2, size=(training_data.shape[0], 1))

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=training_data.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model with the training data
model.fit(training_data, labels, epochs=epochs, batch_size=num_geotiffs,shuffle=True)
