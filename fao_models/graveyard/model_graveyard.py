# Define your TensorFlow models here
def cnn_v1_softmax_onehot(optimizer,loss_fn,metrics=['accuracy']):
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

    
    inputs = layers.Input(shape=[None, None, 4])
    encoder0_pool, encoder0 = encoder_block(inputs, 16)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 64)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 128)
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 256)
    center = conv_block(encoder4_pool, 512)

    outputs = layers.Dense(2, activation='softmax')(center) # 2 values output
    outputs = tf.squeeze(outputs) # Kyle Dormans suggestion
    
    model = models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)

    return model

def simple_model_sigmoid_onehot(optimizer,loss_fn,metrics=['accuracy']):
    model = models.Sequential([
        layers.Input(shape=(32, 32, 4)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='sigmoid'),
        tf.squeeze(layers) # otherwise am getting ValueError: Shapes (None, 1, 2) and (None, 2) are incompatible
        # but this traceback could be indication of a different issue..?
    ])

    model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)

    return model

def resnet(optimizer,loss_fn,metrics=['accuracy']):
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
        
        model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)
        
        return model
    return create_resnet_with_4_channels(input_shape=(32, 32, 4), num_classes=1)