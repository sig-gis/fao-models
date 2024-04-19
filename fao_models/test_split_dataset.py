import dataloader as dl
data_dir = r"C:\fao-models\tfrecords\balanced"
batch_size = 10
total_examples = 1200
test_split = 0.2
val_split = 0.1
buffer_size = total_examples
# Load the dataset without batching
dataset = dl.load_dataset_from_tfrecords(data_dir, batch_size=batch_size)
print(dataset)

# Split the dataset into training and testing
# train should have 80 examples, test have 20
train_dataset, test_dataset = dl.split_dataset(
    dataset, total_examples, test_split=test_split, batch_size=batch_size
)

train_examples = list(train_dataset.unbatch().as_numpy_iterator())
test_examples = list(test_dataset.unbatch().as_numpy_iterator())

print('Previous 2-way split')
print(len(train_examples))
print(len(test_examples))
print(len(train_examples) + len(test_examples) == total_examples)


train_dataset, test_dataset, val_dataset = dl.split_dataset(
    dataset,total_examples,test_split=test_split,val_split=val_split,batch_size=batch_size)  

train_examples = list(train_dataset.unbatch().as_numpy_iterator())
test_examples = list(test_dataset.unbatch().as_numpy_iterator())
val_examples = list(val_dataset.unbatch().as_numpy_iterator())

print('New 3-way split')
print(len(train_examples))
print(len(test_examples))
print(len(val_examples))
print(len(train_examples) + len(test_examples) + len(val_examples) == total_examples)
