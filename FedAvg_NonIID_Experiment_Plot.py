import torch
import matplotlib.pyplot as plt
import numpy as np
from mnist_dataloader import MNISTDataloader
from torchvision import datasets, transforms
from server import Server

np.random.seed(42)
torch.manual_seed(42)

configurations = [
    {"batch_size": 10, "epochs": 1},
    {"batch_size": 10, "epochs": 5},
    {"batch_size": 10, "epochs": 20},
    {"batch_size": 50, "epochs": 1},
    {"batch_size": 50, "epochs": 5},
    {"batch_size": 50, "epochs": 20},
    {"batch_size": float("inf"), "epochs": 1},
    {"batch_size": float("inf"), "epochs": 5},
    {"batch_size": float("inf"), "epochs": 20},
]

# Storing test accuracy results
results = {}

# 60,000 images
transform = transforms.Compose([transforms.ToTensor()])
train_mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Store results num_rounds times
def run_non_iid_plotting(num_rounds=750):
    num_clients = 100
    examples_per_client = len(train_mnist_data) // num_clients
    shard_size = examples_per_client // 2 # two shards per client

    train_dataloader_non_iid = MNISTDataloader(
        dataset=train_mnist_data, 
        num_clients=num_clients, 
        shard_size=shard_size, 
        is_iid=False
    )

    for config in configurations:
        batch_size = config["batch_size"]
        epochs = config["epochs"]

        server = Server(
            server_id=0,
            train_dataloader_iid=None,
            train_dataloader_non_iid=train_dataloader_non_iid,
            num_clients=num_clients,
            is_iid=False,
            num_rounds=num_rounds
        )

        # modify the model's local training parameters
        server.model_constants["LOCAL_MINIBATCH"] = min(batch_size, len(train_mnist_data)) if batch_size != float("inf") else len(train_mnist_data)
        server.model_constants["LOCAL_EPOCHS"] = epochs

        server.start()

        # store results from dis pairing
        test_accuracies = server.test_accuracies
        results[(batch_size, epochs)] = test_accuracies
        

run_non_iid_plotting()

# Plotting code - brought to you the one and only Chat
plt.figure(figsize=(8, 6))
colors = ["red", "orange", "blue"]  # Colors for different batch sizes
linestyles = ["-", "--", ":"]  # Linestyles for different epochs

for i, config in enumerate(configurations):
    batch_size, epochs = config["batch_size"], config["epochs"]
    label = f"B={batch_size if batch_size != float('inf') else '∞'} E={epochs}"
    color = colors[i // 3]  # Switch color every 3 configs cuz there's 3 types
    linestyle = linestyles[i % 3]  # There's 3 types as well for each type of batch size
    plt.plot(results[(batch_size, epochs)], label=label, color=color, linestyle=linestyle)

# Graph formatting
plt.xlabel("Communication Rounds")
plt.ylabel("Test Accuracy")
plt.title("MNIST CNN Non-IID")
plt.legend()
plt.grid(True)
plt.show()
