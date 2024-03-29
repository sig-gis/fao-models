import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as backend
from tensorflow.keras.metrics import categorical_accuracy

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

def model1(optimizer,loss_fn,metrics=['accuracy']):
    model = models.Sequential([
        layers.Input(shape=(32, 32, 4)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='softmax')
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

# Create a dictionary of keyword-function pairs
model_dict = {
    'model1': model1,
    'resnet': resnet,
    # in graveyard
    # 'cnn_v1_softmax_onehot': cnn_v1_softmax_onehot,
    # 'simple_model_sigmoid_onehot': simple_model_sigmoid_onehot,
}

def get_model(model_name,**kwargs):
    if model_name in model_dict:
        print(f"Model found: {model_name}")
        model_function = model_dict[model_name]
        # model_function()
    else:
        print(f"Model '{model_name}' not found.")

    return model_function(**kwargs)

if __name__ == "__main__":
    # Example usage
    model_name = 'resnet'
    optimizer = 'adam'
    loss_fn = 'binary_crossentropy'
    metrics = ['accuracy']
    m = get_model(model_name,optimizer=optimizer,loss_fn=loss_fn,metrics=metrics)
    m.summary()