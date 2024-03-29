#%%
import datetime
import logging
from models import get_model
import dataloader as dl
import os
import tensorflow as tf
import yaml
from pprint import pformat

# TODO: make this single CLI arg input
config_file = r"C:\fao-models\runc2.yml"

# setup logging 
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.WARNING,
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          f'trainlog_{datetime.datetime.now().strftime("%Y-%m-%d")}.log') # add _%H-%M-%S if needbe
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open(config_file, 'r') as file:
    config_data = yaml.safe_load(file)

# retrieve parameters
model_name = config_data['model_name']
total_examples = config_data['total_examples']
data_dir = config_data['data_dir']
data_split = config_data['data_split']
epochs = config_data['epochs']
learning_rate = config_data['learning_rate']
batch_size = config_data['batch_size']
buffer_size = config_data['buffer_size']
optimizer = config_data['optimizer']
loss_function = config_data['loss_function']

# hyperbolically decrease the learning rate to 1/2 of the base rate at 1,000 epochs, 1/3 at 2,000 epochs, and so on.
if optimizer == 'adam':
    steps_per_epoch = total_examples*data_split // batch_size
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=learning_rate,
    decay_steps=steps_per_epoch*epochs,
    decay_rate=1,
    staircase=False)
    logger.info(f'Using a learning rate schedule of InverseTimeDecay, decay_steps={steps_per_epoch*epochs}')

    optimizer = tf.keras.optimizers.Adam(lr_schedule)

# pull model from config
model = get_model(model_name,optimizer=optimizer,loss_fn=loss_function)       
print(model.summary())

logger.info('Config file: %s', config_file)
logger.info('Parameters:')
logger.info(pformat(config_data))

# Load the dataset without batching
dataset = dl.load_dataset_from_tfrecords(data_dir)

# Split the dataset into training and testing
train_dataset, test_dataset = dl.split_dataset(dataset, total_examples, test_split=data_split, batch_size=batch_size)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

logger.info('Starting model training...')
LOGS_DIR = '.\logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[early_stop,tf.keras.callbacks.TensorBoard(LOGS_DIR)])

logger.info('Model training complete')
logger.info('Training history:')
logger.info(pformat(history.history))