import torch
from torchvision import datasets, transforms
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import numpy as np


transform = transforms.Compose([transforms.ToTensor()])
# # 60,000 pictures
train_mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# # 10,000 pictures
test_mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)



# Splits indices into training and validation sets
def split_indices(num_samples, val_ratio=0.2):
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices

# Create IID Split
def create_iid_split(data, num_clients=100):
    num_samples = len(data)
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    client_data = np.array_split(indices, num_clients)

    return client_data

# Create Non-IID Split
def create_non_iid_split(data, num_clients=100, shards_per_client=2, shard_size=300):
    num_classes = 10
    class_indices = defaultdict(list)

    # Sort indices by digit label
    for idx, (_, label) in enumerate(data):
        class_indices[label].append(idx)

    # print(class_indices.keys())

    # Create shards
    shards = []
    for label in range(num_classes):
        np.random.shuffle(class_indices[label])  # Shuffle indices of each class
        shard_size = min(len(class_indices[label]) - 1, shard_size)
        for i in range(0, len(class_indices[label]), shard_size):
            shards.append(class_indices[label][i:i + shard_size])

    np.random.shuffle(shards)  # Shuffle the shards

    # Assign shards to clients
    client_data = {i: [] for i in range(num_clients)}
    for i in range(num_clients):
        client_data[i].extend(shards[i * shards_per_client:(i + 1) * shards_per_client])

    return client_data

class MNISTDataloader(DataLoader):
    def __init__(self, dataset, num_clients=100, val_ratio=0.0, shard_size=300, is_iid=True):
        self.dataset = dataset
        self.num_clients = num_clients
        self.val_ratio = val_ratio
        self.shard_size = shard_size
        self.is_iid = is_iid

        # Split dataset into train and validation sets
        self.train_indices, self.val_indices = split_indices(len(self.dataset), self.val_ratio)
        # self.train_indices, self.val_indices = split_indices(1000, self.val_ratio)
        self.train_data = [(self.dataset[i][0], self.dataset[i][1]) for i in self.train_indices]
        self.val_data = [(self.dataset[i][0], self.dataset[i][1]) for i in self.val_indices]

        # Create splits for federated learning
        if self.is_iid:
            self.client_data = create_iid_split(self.train_data, self.num_clients)
        else:
            self.client_data = create_non_iid_split(self.train_data, self.num_clients, shard_size=self.shard_size)

    # Returns the data assigned to a particular client
    def get_client_data(self, client_id):
        return self.client_data[client_id]

    # Returns the full training data
    def get_train_data(self):
        return self.train_data

    # Returns the validation data
    def get_val_data(self):
        return self.val_data

    # Returns a DataLoader for a particular client
    def get_dataloader(self, client_id, batch_size=32, shuffle=True):
        client_indices = self.client_data[client_id]
        if not self.is_iid:
          client_indices = [item for shard in client_indices for item in shard]
        client_dataset = torch.utils.data.Subset(self.train_data, client_indices)
        return DataLoader(client_dataset, batch_size=batch_size, shuffle=shuffle)

    # Returns a DataLoader for the full training data
    def get_full_train_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle)

    # Returns a DataLoader for the validation data
    def get_val_dataloader(self, batch_size=32, shuffle=False):
        return DataLoader(self.val_data, batch_size=batch_size, shuffle=shuffle)


# num_clients = 20
# batch_size = 32
# dataloader = MNISTDataloader(dataset=train_mnist_data, num_clients=num_clients, is_iid=True)

# # Get a DataLoader for client 0
# client_0_dataloader = dataloader.get_dataloader(client_id=0, batch_size=batch_size)

# # Get validation DataLoader
# val_dataloader = dataloader.get_val_dataloader(batch_size=batch_size)

    