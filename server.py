from client import Client
from models import MNISTCNN
import numpy as np
import threading
from collections import OrderedDict
import torch
import copy
from mnist_dataloader import MNISTDataloader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from sklearn.cluster import SpectralClustering
import math


# import pdb
# pdb.set_trace()

# Define constants for federated learning
model_constants = {
    "CLIENT_FRACTION": 0.1,
    "LOCAL_MINIBATCH": 10,
    "LOCAL_EPOCHS": 5,
    "LEARNING_RATE": 0.01,
    "LABELS_PER_CLIENT": 2,
    "COMMUNICATION_ROUNDS": 100,
    "NUM_CLUSTERS": 3
}

class Server():
    def __init__(self, train_dataloader_iid, train_dataloader_non_iid, num_clients, is_iid=True, num_rounds=model_constants["COMMUNICATION_ROUNDS"], clients_ids=[], client_fraction=model_constants["CLIENT_FRACTION"]):
        self.client_ids = clients_ids
        self.clients_per_round: int = math.ceil(client_fraction * len(self.client_ids))
        self.batch_size = model_constants["LOCAL_MINIBATCH"]

        self.server_cv: threading.Condition = threading.Condition(threading.Lock())

        # list: index is client id and value is tuple of (weights, loss)
        self.clients_training_data: list[tuple] = [(0,0)] * num_clients
        self.devices_done_running: set = set()

        self.global_model: MNISTCNN = MNISTCNN(model_constants)

        self.num_rounds = num_rounds

        if is_iid:
            train_dataloader=train_dataloader_iid
        else:
            train_dataloader=train_dataloader_non_iid

        print("Done splitting the data.")
        self.clients = OrderedDict()
        for client_id in range(num_clients):
            new_client = Client(
                client_id=client_id,
                server=self,
                train_dataloader=train_dataloader.get_dataloader(client_id=client_id, batch_size=self.batch_size)
            )
            
            self.clients[client_id] = new_client


    def get_server_cv(self):
        return self.server_cv


    def send_client_result(self, client_id: int, client_result: tuple):
        self.clients_training_data[client_id] = client_result
        self.devices_done_running.add(client_id)
    
    def convergence_criteria(self):
        # Note: could experiment with stopping when the model reaches a certain accuracy
        self.num_rounds -= 1
        if self.num_rounds < 0:
            return True
        return False


    def start(self):
        # Main training loop
        self.server_cv.acquire()
        # breakpoint()
        while not self.convergence_criteria():
            print("\n\n########################################################################\n", end="")
            print(f"SERVER INITIALIZING COMMUNICATION ROUND #{100 - self.num_rounds}\n", end="")
            print("########################################################################\n\n\n", end="")
            devices_to_run = np.random.choice(self.client_ids, size=self.clients_per_round, replace=False)
            self.devices_done_running.clear()

            # Signal devices to run
            for device in devices_to_run:
                self.clients[device].send_global_model(copy.deepcopy(self.global_model))
                self.clients[device].run_training()

            # Wait for devices to be done running
            while len(self.devices_done_running) != len(devices_to_run):
                self.server_cv.wait()

            # Update global model to reflect client updates
            avg_model = self.fed_avg(clients=devices_to_run)
            self.global_model.load_state_dict(avg_model)

            # Test global model performance
            self.test_model()

        self.server_cv.release()

        # Signal all devices to exit and return when they are done
        for i in self.client_ids:
            self.clients[i].kill()
            self.clients[i].join()
        
        return self.clients_training_data


    # Return the state dictionary of the global model after training
    def fed_avg(self, clients: np.array, weight_log = False):
        weights = np.array([len(self.clients[cli].train_dataloader) if cli in clients else 0 for cli in self.clients])
        weights = weights / np.sum(weights)

        # take log of weights if testing extension 2
        if weight_log:
            weights = np.log(weights)

        # Get the state dict of the first client that returned
        first_state_dict = self.clients_training_data[clients[0]][0]

        avg_weights = OrderedDict()
        for key in first_state_dict.keys():
            curr_weights = [weights[i] * self.clients_training_data[i][0][key] for i in clients]
            stacked_weights = torch.stack(curr_weights)
            avg_weights[key] = torch.sum(stacked_weights, dim=0)
            # print(f"Weights {key}: {avg_weights[key]}\n", end="")

        return avg_weights


    # EXTENSION 2: 
    def fed_avg_log(self):
        self.fed_avg(weight_log=True)


    def test_model(self):
        test_mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = DataLoader(test_mnist_data)
        # Set the model to evaluation mode
        self.global_model.eval()

        # Initialize lists to hold predictions and ground-truth labels
        all_predictions = []
        all_labels = []

        # Disable gradient calculation for testing (increases efficiency)
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move inputs and labels to the appropriate device (GPU or CPU)
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
                
                # Get model predictions
                outputs = self.global_model(inputs)
                
                # get model prediction
                _, predictions = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy
        accuracy = sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
        print(f'Accuracy: {accuracy * 100:.2f}%')

