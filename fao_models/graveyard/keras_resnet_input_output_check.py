#%%
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
# The inputs is a batch of 10 32x32 RGBN images with 4 channels
input_shape = (10, 32, 32, 4)
num_classes = 1
x = tf.random.normal(input_shape)
print(x.shape)
print(x)
# %%
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
    # x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)  # Use 'softmax' for multi-class

    # Create the final model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Change to 'categorical_crossentropy' for multi-class
        metrics=['accuracy'], 
        
    )

    return model

# Example usage:
model = create_resnet_with_4_channels(input_shape=(32, 32, 4), num_classes=1)
predictions = model(x).numpy()
print(predictions)
# %%
