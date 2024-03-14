#%%
import tensorflow as tf
import tensorflow.keras.layers as layers
# The inputs is a batch of 10 32x32 RGBN images with 4 channels
input_shape = (10, 32, 32, 4)
x = tf.random.normal(input_shape)
# %%
# example from Ates crop mapping CNN 
# model architecture will consist of encoder blocks which consists of a conv_block then maxPooling2D
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
#%%
# demonstrating what happens in the encoders...
print('original shape',x.shape) # start with a batch of 10 32x32x4 arrays
# inputs = layers.Input(shape=[None, None, 4])
encoder0_pool, encoder0 = encoder_block(x, 16)
print('encoder0_pool.shape',encoder0_pool.shape)
print(encoder0_pool[0][0])
print('encoder0.shape',encoder0.shape)
print(encoder0[0][0])
# %%
# pass that result thru another encoder block
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)
print('encoder1_pool.shape',encoder1_pool.shape)
print(encoder1_pool[0][0])
print('encoder1.shape',encoder1.shape)
print(encoder1[0][0])
# %%
# do this a few more times till we get to the center
encoder2_pool, encoder2 = encoder_block(encoder1_pool, 64)
encoder3_pool, encoder3 = encoder_block(encoder2_pool, 128)
encoder4_pool, encoder4 = encoder_block(encoder3_pool, 256)
center = conv_block(encoder4_pool, 512)

# eventually get center which is 10 1D arays of 512 values
print('center.shape',center.shape) # (10, 1, 1, 512) 
print(center[0][0])
# %%
# and then pass to output
# this is the output in Ates model arch, this is 1D arrays with 2 channels, 
outputs_num_filter2 = layers.Conv2D(2, (1, 1), activation='sigmoid')(center)
print('outputs_num_filter2.shape',outputs_num_filter2.shape) # (10, 1, 1, 2)
print(outputs_num_filter2[0][0])

# wouldn't we want just one value? (10, 1, 1, 1)? 
outputs_num_filter1 = layers.Conv2D(1, (1, 1), activation='sigmoid')(center)
print('outputs_num_filter1.shape',outputs_num_filter1.shape) # (10, 1, 1, 1)
print(outputs_num_filter1[0][0])

# %%
# and then how do you get a 1 or 0, not floating point output? 
# but I'm assuming that you don't want to have this as the output in the compiled model because you can't validate assess loss, 
# rather would the model output the floating point value and then you'd use a threshold to convert to 1 or 0 when serving inferences?
output_class = tf.cast(tf.greater(outputs_num_filter1, 0.5), dtype=tf.int32)
print('output_class',output_class[0][0])
# %%
