import torch
from torchvision import datasets, transforms
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random

# or any number of clients
N_CLIENTS = 16  
EXAMPLES_PER_CLIENT = 3750
LABELS_PER_CLIENT = 2

transform = transforms.Compose([transforms.ToTensor()])
# 60,000 pictures
train_mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 10,000 pictures
test_mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Divide each label group into chunks of unique examples
label_data = defaultdict(list)
for idx, (_, label) in enumerate(train_mnist_data):
    label_data[label].append(idx)

client_data_indices = []
examples_per_label_group = EXAMPLES_PER_CLIENT // LABELS_PER_CLIENT

for client in range(N_CLIENTS):

    chosen_labels = random.sample(range(10), labels_per_client)
    client_indices = []

    for label in chosen_labels:
        selected_indices = label_data[label][:examples_per_label_group]
        label_data[label] = label_data[label][examples_per_label_group:]
        client_indices.extend(selected_indices)

    client_data_indices.append(client_indices)




#abhi's stuff
# Create IID Split
def create_iid_split(data, num_clients=100, examples_per_client=600):
    num_samples = len(data)
    indices = np.random.permutation(num_samples)
    client_data = {i: [] for i in range(num_clients)}
    
    for i in range(num_samples):
        client_id = i // examples_per_client
        client_data[client_id].append(indices[i])

    return client_data

# Create Non-IID Split
def create_non_iid_split(data, num_clients=100, shards_per_client=2, shard_size=300):
    num_classes = 10
    class_indices = defaultdict(list)

    # Sort indices by digit label
    for idx, (_, label) in enumerate(data):
        class_indices[label].append(idx)

    # Create shards
    shards = []
    for label in range(num_classes):
        np.random.shuffle(class_indices[label])  # Shuffle indices of each class
        for i in range(0, len(class_indices[label]), shard_size):
            shards.append(class_indices[label][i:i + shard_size])

    np.random.shuffle(shards)  # Shuffle the shards

    # Assign shards to clients
    client_data = {i: [] for i in range(num_clients)}
    for i in range(num_clients):
        client_data[i].extend(shards[i * shards_per_client:(i + 1) * shards_per_client])

    return client_data

# Generate splits
train_indices, val_indices = split_indices(len(mnist_data))
train_data = [(mnist_data[i][0], mnist_data[i][1]) for i in train_indices]
val_data = [(mnist_data[i][0], mnist_data[i][1]) for i in val_indices]

# Create splits for training data
iid_train_split = create_iid_split(train_data)
non_iid_train_split = create_non_iid_split(train_data)
