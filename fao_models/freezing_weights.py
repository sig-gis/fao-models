#%%
import keras
from models import get_model
import dataloader as dl
import numpy as np

#%%
dir_path = "C:\\fao-models\\tfrecords\\all"
dataset = dl.load_dataset_from_tfrecords(tfrecord_dir=dir_path, batch_size=32, buffer_size=1000, seed=5)
train_dataset, test_dataset, val_dataset = dl.split_dataset(dataset, total_examples=2000, test_split=0.2, batch_size=32, val_split=0.1)

# y_true = np.concatenate([y for x, y in dataset], axis=0)
# print('y_true count: ',len(y_true))
# vals, counts = np.unique(y_true, return_counts=True)
# print('vals, counts: ',[vals, counts])

# %%
# load model
model = get_model(model_name="resnet", optimizer="adam", loss_fn="binary_crossentropy")
# model.summary()
# #%%
# # train the model
# history = model.fit(
#         train_dataset,
#         epochs=1,
#         validation_data=test_dataset,
#         # callbacks=callbacks,
#     )
#%%

def freeze(model):
    """Freeze model weights in every layer."""
    for layer in model.layers:
        layer.trainable = False

        if isinstance(layer, keras.models.Model):
            freeze(layer)
    return model

def unfreeze(model):
    """Unfreeze model weights in every layer."""
    for layer in model.layers:
        layer.trainable = True

        if isinstance(layer, keras.models.Model):
            unfreeze(layer)
    return model

normal_weights_file = "C:\\fao-models\\saved_models\\test-freezing-weights\\model.h5"
frozen_weights_file = "C:\\fao-models\\saved_models\\test-freezing-weights\\frozen_model.h5"
#%%
# save model as-is, no freezing
# model.save_weights(normal_weights_file)

# freeze model then save
# frozen = freeze(model)
# frozen.save_weights(frozen_weights_file)

# load model checkpoint from normal weights and from frozen see if any differneces
# loading non-frozen weights into non-frozen model ok 
model1 = model2 = model
model1.load_weights(normal_weights_file)

# if you don't freeze model before loading weights of a frozen model, get axes don't match array error
freeze(model2)
model2.load_weights(frozen_weights_file)

#%%
model1.evaluate(val_dataset)
model2.evaluate(val_dataset)
#%%
preds_model1 = model1.predict(val_dataset)[0:20]
preds_model2 = model2.predict(val_dataset)[0:20]
assert np.array_equal(preds_model1, preds_model2), "Predictions are not equal"
#%%
# load weights from trainable model
model3 = get_model(model_name="resnet", optimizer="adam", loss_fn="binary_crossentropy")
model3.load_weights(normal_weights_file)

# then freeze the model for inference
freeze(model3)
model3.evaluate(val_dataset)
preds_model3_frozen = model3.predict(val_dataset)[0:20]

# try unfreezing and training it again
unfreeze(model3)
model3.evaluate(val_dataset)
preds_model3_unfrozen = model3.predict(val_dataset)[0:20]

assert np.array_equal(preds_model3_frozen, preds_model3_unfrozen), "Predictions are not equal"
print(preds_model3_frozen)
print(preds_model3_unfrozen)

#%%


# # in another script or workflow...
# new_model_unfrozen = get_model(model_name="resnet", optimizer="adam", loss_fn="binary_crossentropy")
# new_model_unfrozen.load_weights(normal_weights_file) # try to load weights

# new_model_frozen = get_model(model_name="resnet", optimizer="adam", loss_fn="binary_crossentropy")
# new_model_frozen.load_weights(frozen_weights_file) # try to load weights
# %%
# print(new_model_unfrozen.predict(val_dataset.take(1)))
# %%
