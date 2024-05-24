# %%
# % load_ext autoreload
# %autoreload `2
import dataloader as dl
import numpy as np

data_dir = "tfrecords/all"
batch_size = 64
buffer_size = 135232
total_examples = buffer_size
seed = 5
test_split = 0.2
val_split = 0.1

# %%
dataset = dl.load_dataset_from_tfrecords(
    data_dir, batch_size=batch_size, buffer_size=buffer_size, seed=seed, shuffle=False
)

train_dataset, test_dataset, val_dataset = dl.split_dataset(
    dataset,
    total_examples,
    test_split=test_split,
    batch_size=batch_size,
    val_split=val_split,
)
# %%
print("Reporting class balance for each data split...")

print("All Data")
y_true = np.concatenate([y for x, y in dataset], axis=0)
print("y_true count: ", len(y_true))
vals, counts = np.unique(y_true, return_counts=True)
print("vals, counts: ", [vals, counts])

print("Train Data")
y_true_train = np.concatenate([y for x, y in train_dataset], axis=0)
print("y_true count: ", len(y_true_train))
vals, counts = np.unique(y_true_train, return_counts=True)
print("vals, counts: ", [vals, counts])

print("Test Data")
y_true_test = np.concatenate([y for x, y in test_dataset], axis=0)
print("y_true count: ", len(y_true_test))
vals, counts = np.unique(y_true_test, return_counts=True)
print("vals, counts: ", [vals, counts])

print("Val Data")
y_true_val = np.concatenate([y for x, y in val_dataset], axis=0)
print("y_true count: ", len(y_true_val))
vals, counts = np.unique(y_true_val, return_counts=True)
print("vals, counts: ", [vals, counts])
# %%
from models import get_model, freeze
from pprint import pformat
import numpy as np

checkpoint = "expriments/resnet-epochs20-batch64-lr001-seed5-lrdecay5-tfrecords-all/best_model.h5"
# checkpoint = "C:\\fao-models\\saved_models\\mobilenet_v3small_batch255\\best_model.h5"
model = get_model(
    model_name="resnet",
    optimizer="adam",
    loss_fn="binary_crossentropy",
    training_mode=True,
)
model.load_weights(checkpoint)
freeze(model)  # freeze weights for inference

model2 = get_model(
    model_name="resnet",
    optimizer="adam",
    loss_fn="binary_crossentropy",
    training_mode=False,
)
model2.load_weights(checkpoint)
freeze(model2)  # freeze weights for inference
# %%
# use model.evaluate() to get the metrics we want to validate..
eval = model.evaluate(val_dataset, return_dict=True)
print(f"Validation: {pformat(eval)}")
eval2 = model2.evaluate(val_dataset, return_dict=True)
print(f"Validation: {pformat(eval2)}")
# %%
# testing that the values of model.predict() are the same for the same every time, they are not..
pred1 = np.round(model.predict(val_dataset), 1)
pred2 = np.round(model.predict(val_dataset), 1)
print(np.array_equal(pred1, pred2))
# the values aren't even close which makes me think they are different data examples
print(pred1[:10], pred2[:10])
# # %%
# create y_true, y_pred for actual validation dataset
y_true_val = np.concatenate([y for x, y in val_dataset], axis=0)
y_pred_val = model.predict(val_dataset)
print(y_true_val)
print(y_pred_val)


# # %%
def precision(y_true, y_pred):
    """Precision = TP / (TP + FP)"""
    y_true = np.rint(y_true)
    y_pred = np.ravel(np.rint(y_pred))
    tp = np.sum(np.where((y_true == 1) & (y_pred == 1), 1, 0))
    fp = np.sum(np.where((y_true == 0) & (y_pred == 1), 1, 0))
    return tp / (tp + fp)


# test
y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0])

tp = np.sum(np.where((y_true == 1) & (y_pred == 1), 1, 0))
fp = np.sum(np.where((y_true == 0) & (y_pred == 1), 1, 0))
pr = tp / (tp + fp)
print("dummy data precision", pr)
# %%
# after verifying math checks out, yikes huge difference
print("custom precision fn", precision(y_true_val, y_pred_val))
print("eval result", eval.get("precision"))
# print('y_true, y_pred')
# for i in list(zip(y_true_val[:10],np.round(y_pred_val)[:10])):
#     print(i)


# %%
# binary accuracy
def binary_accuracy(y_true, y_pred):
    """Accuracy = Agreements / Total examples"""
    y_true = np.rint(y_true)
    y_pred = np.ravel(np.rint(y_pred))  # to 1D array
    agreement = np.sum(np.where(y_true == y_pred, 1, 0))
    return agreement / np.size(y_true)


# test on dummy data
y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0])
binary_acc = binary_accuracy(y_true, y_pred)
print("dummy data bin acc", binary_acc)

# after verifying mathc checks out,
print("custom bin acc", binary_accuracy(y_true_val, y_pred_val))
print("eval result", eval.get("binary_accuracy"))
# print('y_true, y_pred')
# for i in list(zip(y_true_val[:10],np.round(y_pred_val)[:10])):
#     print(i)


# %%
def recall(y_true, y_pred):
    """Recall = TP / (TP + FN)"""
    y_true = np.rint(y_true)
    y_pred = np.ravel(np.rint(y_pred))  # to 1D array
    tp = np.sum(np.where((y_true == 1) & (y_pred == 1), 1, 0))
    fn = np.sum(np.where((y_true == 1) & (y_pred == 0), 1, 0))
    return tp / (tp + fn)


# test on dummy data
y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0])
recall_score = recall(y_true, y_pred)
print("dummy data recall", recall_score)

# after verifying mathc checks out,
print("custom recall result", recall(y_true_val, y_pred_val))
print("eval result", eval.get("recall"))
# print('y_true, y_pred')
# for i in list(zip(y_true_val[:10],np.round(y_pred_val)[:10])):
#     print(i)
