import dataloader as dl
import numpy as np
data_dir = "C:\\fao-models\\tfrecords\\all"
batch_size=64
buffer_size=76992
total_examples=buffer_size
test_split=0.2
val_split=0.1

# Load the dataset without batching
dataset = dl.load_dataset_from_tfrecords(data_dir, batch_size=batch_size, buffer_size=buffer_size, seed=5)

# Split the dataset 2 ways or 3 ways
if val_split is not None:
    train_dataset, test_dataset, val_dataset = dl.split_dataset(
        dataset,
        total_examples,
        test_split=test_split,
        batch_size=batch_size,
        val_split=val_split,
    )
    
else:
    train_dataset, test_dataset = dl.split_dataset(
    dataset, total_examples, test_split=test_split, batch_size=batch_size
    )
    
# # checking data splits for class balance
# print('Reporting class balance for each data split...')

# print('All Data')
# y_true = np.concatenate([y for x, y in dataset], axis=0)
# print('y_true count: ',len(y_true))
# vals, counts = np.unique(y_true, return_counts=True)
# print('vals, counts: ',[vals, counts])

# print('Train Data')
# y_true_train = np.concatenate([y for x, y in train_dataset], axis=0)
# print('y_true count: ',len(y_true_train))
# vals, counts = np.unique(y_true_train, return_counts=True)
# print('vals, counts: ',[vals, counts])

# print('Test Data')
# y_true_test = np.concatenate([y for x, y in test_dataset], axis=0)
# print('y_true count: ',len(y_true_test))
# vals, counts = np.unique(y_true_test, return_counts=True)
# print('vals, counts: ',[vals, counts])

# print('Val Data')
# y_true_val = np.concatenate([y for x, y in val_dataset], axis=0)
# print('y_true count: ',len(y_true_val))
# vals, counts = np.unique(y_true_val, return_counts=True)
# print('vals, counts: ',[vals, counts])

# train_dataset = train_dataset.shuffle(
#     buffer_size, reshuffle_each_iteration=True)

# inspect each batch to ensure it is balanced
for batch_images, batch_labels in val_dataset:
    vals, counts = np.unique(batch_labels, return_counts=True)
    print('vals, counts: ', [vals, counts])
