#%%
import keras
from models import get_model
import dataloader as dl
import numpy as np

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

#%%
dir_path = "C:\\fao-models\\tfrecords\\all"
dataset = dl.load_dataset_from_tfrecords(tfrecord_dir=dir_path, batch_size=32, buffer_size=140000, seed=5)
train_dataset, test_dataset, val_dataset = dl.split_dataset(dataset, total_examples=2000, test_split=0.2, batch_size=32, val_split=0.1)

y_true = np.concatenate([y for x, y in val_dataset], axis=0)
print('y_true count: ',len(y_true))
vals, counts = np.unique(y_true, return_counts=True)
print('vals, counts: ',[vals, counts])
# %%
# load model 
weights_file = "C:\\fao-models\\saved_models\\resnet-epochs10-batch64-lr001-seed5-lrdecay5-tfrecords-all\\best_model.h5"

# this is how we've had it before, base_model(training=True) when building top-layers on top of the resent
model_trainable = get_model(model_name="resnet", optimizer="adam", loss_fn="binary_crossentropy", training_mode=True)
# this would be used if you don't want to train the base model as well (we do)
model_nontrainable = get_model(model_name="resnet", optimizer="adam", loss_fn="binary_crossentropy", training_mode=False)

model_trainable.load_weights(weights_file)
model_nontrainable.load_weights(weights_file)

print(model_trainable.summary())
print(model_nontrainable.summary())

#%%
# check if trainable weights, non-trainable weights, and model predictions differ
model_trainable.evaluate(val_dataset)
model_trainable_preds = np.round(model_trainable.predict(val_dataset),1)[0:20]

model_nontrainable.evaluate(val_dataset)
model_nontrainable_preds = np.round(model_nontrainable.predict(val_dataset),1)[0:20]

# the trainable/non-trainable weights count are the same between them but the actual weights (np arrays) 
# are different so this is not the same thing as freezing weights (model.trainable=False)
assert np.array_equal(len(model_trainable.trainable_weights),len(model_nontrainable.trainable_weights)), "Trainable weights not equal"
assert np.array_equal(len(model_trainable.non_trainable_weights),len(model_nontrainable.non_trainable_weights)), "Non-Trainable weights not equal"
# but predictions are not equal which we expected
assert np.array_equal(model_trainable_preds,model_nontrainable_preds), "Predictions not equal"

#%%
# load model and freeze all layer weights (layer.trainable=False)
model_frozen = get_model(model_name="resnet", optimizer="adam", loss_fn="binary_crossentropy")
model_frozen.load_weights(weights_file)
freeze(model_frozen)
model_frozen.evaluate(val_dataset)
model_frozen_preds = np.round(model_frozen.predict(val_dataset),1)[0:20]

# don't freeeze weights, just load
model_unfrozen = get_model(model_name="resnet", optimizer="adam", loss_fn="binary_crossentropy")
model_unfrozen.load_weights(weights_file)
model_unfrozen.evaluate(val_dataset)
model_unfrozen_preds = np.round(model_unfrozen.predict(val_dataset),1)[0:20]

assert np.array_equal(len(model_frozen.trainable_weights),len(model_unfrozen.trainable_weights)), "Trainable weights not equal"
assert np.array_equal(len(model_frozen.non_trainable_weights),len(model_unfrozen.non_trainable_weights)), "Non-Trainable weights not equal"
# predictions are not equal
assert np.array_equal(model_frozen_preds,model_unfrozen_preds), "Predictions not equal"
# %%
# interestingly.. you can still train the frozen model. not sure whats actually happening if all weights are frozen
model_frozen.fit(
        train_dataset,
        epochs=1,
        validation_data=test_dataset,
        # callbacks=callbacks,
    )
# %%
