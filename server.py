from client import Client
from models import MNISTCNN, model_constants
import numpy as np
import threading
from collections import OrderedDict
import torch
import copy
from mnist_dataloader import MNISTDataloader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
# import pdb
# pdb.set_trace()

# Define constants for federated learning
model_constants = {
    "NUM_DEVICES": 100,
    "DEVICES_PER_EPOCH": 10,
    "LOCAL_MINIBATCH": 10,
    "LOCAL_EPOCHS": 5,
    "LEARNING_RATE": 0.215,
    "LABELS_PER_CLIENT": 2,
    "COMMUNICATION_ROUNDS": 100,
    "NUM_CLUSTERS": 3
}

class Server():
    def __init__(self, num_rounds=model_constants["COMMUNICATION_ROUNDS"], is_iid=True):
        self.num_clients: int = model_constants["NUM_DEVICES"]
        self.batch_size = model_constants["LOCAL_MINIBATCH"]

        self.server_cv: threading.Condition = threading.Condition(threading.Lock())

        self.clients_training_data: list[tuple] = [(0,0)] * self.num_clients
        self.devices_done_running: set = set()

        self.global_model: MNISTCNN = MNISTCNN(model_constants)

        self.num_rounds = num_rounds

        transform = transforms.Compose([transforms.ToTensor()])
        # 60,000 pictures 
        train_mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # 10,000 pictures
        self.test_mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # create dataloaders for each function
        examples_per_client = len(train_mnist_data) / self.num_clients
        shard_size = len(train_mnist_data) / (examples_per_client * 2) # two shards per client
        if is_iid:
            train_dataloader = MNISTDataloader(dataset=train_mnist_data, 
                                                    num_clients = self.num_clients,
                                                    examples_per_client=examples_per_client,
                                                    shard_size=shard_size,
                                                    is_iid=True)
        else:
            train_dataloader = MNISTDataloader(dataset=train_mnist_data, 
                                                    num_clients = self.num_clients,
                                                    examples_per_client=examples_per_client,
                                                    shard_size=shard_size,
                                                    is_iid=False)

        print("Done splitting the data.")
        self.clients = {}
        for client_id in range(self.num_clients):
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
        if self.num_rounds <= 0:
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
            devices_to_run = np.random.choice(range(self.num_clients), size=model_constants["DEVICES_PER_EPOCH"], replace=False)
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
        for i in range(self.num_clients):
            self.clients[i].kill()
            self.clients[i].join()


    # Return the state dictionary of the global model after training
    def fed_avg(self, clients: np.array, weight_log = False):
        weights = np.array([len(cli.train_dataloader) for cli in self.clients.values()])
        weights = weights / np.sum(weights)
        
        # take log of weights if testing extension 2
        if weight_log:
            weights = np.log(weights)

        # Get training data from a random client to get the keys from state_dict
        first_weight = self.clients_training_data[clients[0]][0]
        
        avg_weights = OrderedDict()
        for key in first_weight.keys():
            curr_weights = [weights[i] * self.clients_training_data[i][0][key] for i in clients]
            stacked_weights = torch.stack(curr_weights)
            avg_weights[key] = torch.mean(stacked_weights, dim=0)

        return avg_weights


    # EXTENSION 2: 
    def fed_avg_log(self):
        self.fed_avg(weight_log=True)


    def test_model(self):
        test_loader = DataLoader(self.test_mnist_data)
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

    def run_clustering(self):
        # step 1: run current core model and keep track of weights

        # TODO: run one communication round in the Server

        # step 2: cluster weights to find similar clients
        stored_weights = []
        for client_weights, _ in enumerate(self.clients_training_data):
            #stored_weights.append(np.concatenate([weights.flatten() for weights in client_weights.values()]))

            curr_row = []
            for weights in client_weights.values():
                curr_row.extend(weights.flatten().tolist())
            stored_weights.append(curr_row)

        feature_matrix = np.array(stored_weights)


        clustering = SpectralClustering(n_clusters=model_constants["NUM_CLUSTERS"], assign_labels='discretize', random_state=0).fit(feature_matrix)
        return clustering