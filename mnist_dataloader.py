import torch
from torchvision import datasets, transforms
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import numpy as np
import random
from main import NUM_DEVICES

# Divide each label group into chunks of unique examples
"""label_data = defaultdict(list)
for idx, (_, label) in enumerate(train_mnist_data):
    label_data[label].append(idx)

client_data_indices = []
examples_per_label_group = EXAMPLES_PER_CLIENT // LABELS_PER_CLIENT

for client in range(N_CLIENTS):

    chosen_labels = random.sample(range(10), LABELS_PER_CLIENT)
    client_indices = []

    for label in chosen_labels:
        selected_indices = label_data[label][:examples_per_label_group]
        label_data[label] = label_data[label][examples_per_label_group:]
        client_indices.extend(selected_indices)

    client_data_indices.append(client_indices)"""




#abhi's stuff

def split_indices(num_samples, val_ratio=0.2):
    """Splits indices into training and validation sets."""
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices

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
train_indices, val_indices = split_indices(len(train_mnist_data), 0.15)
train_data = [(train_mnist_data[i][0], train_mnist_data[i][1]) for i in train_indices]
val_data = [(train_mnist_data[i][0], train_mnist_data[i][1]) for i in val_indices]

# Create splits for training data
iid_train_split = create_iid_split(train_data, NUM_DEVICES)
non_iid_train_split = create_non_iid_split(train_data)





class MNISTDataloader():
    def __init__(self, num_clients, batch_size, train_val_split = 0.8):
        # initialize the number of clients
        self.num_clients = num_clients

        # get MNIST data
        self.transform = transforms.Compose([transforms.ToTensor()])
        # 60,000 pictures 
        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # 10,000 pictures
        self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    def get_train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self):
        return self.test_mnist_data
    
    # Create Non-IID Split
    def create_non_iid_split(self, train=True, num_clients=100, shards_per_client=2, shard_size=300):
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
    
    



    

    
    