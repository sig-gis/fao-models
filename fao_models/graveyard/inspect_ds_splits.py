import dataloader as dl
import numpy as np
val_data_dir = r"C:\fao-models\tfrecords\val"
train_test_dir = r"C:\fao-models\tfrecords\train_test"
all_dir = r"C:\fao-models\tfrecords\all"

# load data from train_test directory
# dataset = dl.load_dataset_from_tfrecords(train_test_dir, batch_size=64)
# train,test = dl.split_dataset(dataset, total_examples=7000, test_split=0.2, batch_size=64)
# y_true = np.concatenate([y for x, y in dataset], axis=0)
# print(list(y_true))
# print(len(y_true))
# vals, counts = np.unique(y_true, return_counts=True)
# print(vals, counts)

# load data from val directory - this proves that the last 7 tfrecords didn't have balanced labels
# dataset = dl.load_dataset_from_tfrecords(val_data_dir, batch_size=64)
# y_true = np.concatenate([y for x, y in dataset], axis=0)
# print(list(y_true))
# print(len(y_true))
# vals, counts = np.unique(y_true, return_counts=True)
# print(vals, counts)

# load data from all directory and inspect each of the 3 data splits - this proves that splitting dataset 3 ways in memory is a good way to go 
dataset = dl.load_dataset_from_tfrecords(all_dir, batch_size=64)
print('All')
y_true = np.concatenate([y for x, y in dataset], axis=0)
print(len(y_true))
vals, counts = np.unique(y_true, return_counts=True)
print(vals, counts)

train,test,val = dl.split_dataset(dataset, total_examples=76992, test_split=0.2, batch_size=64, val_split=0.1)

train = train.shuffle(
            buffer_size=76992, reshuffle_each_iteration=True
        )

print('Train')
y_true_train = np.concatenate([y for x, y in train], axis=0)
# print(list(y_true_train))
print(len(y_true_train))
vals, counts = np.unique(y_true_train, return_counts=True)
print(vals, counts)

print('Test')
y_true_test = np.concatenate([y for x, y in test], axis=0)
# print(list(y_true_test))
print(len(y_true_test))
vals, counts = np.unique(y_true_test, return_counts=True)
print(vals, counts)

print('Val')
y_true_val = np.concatenate([y for x, y in val], axis=0)
# print(list(y_true_val))
print(len(y_true_val))
vals, counts = np.unique(y_true_val, return_counts=True)
print(vals, counts)