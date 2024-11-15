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
import pathlib
import glob


# import pdb
# pdb.set_trace()

# Define constants for federated learning
model_constants = {
    "CLIENT_FRACTION": 0.1,
    "LOCAL_MINIBATCH": 10,
    "LOCAL_EPOCHS": 5,
    "LEARNING_RATE": 0.01,
    "LABELS_PER_CLIENT": 2,
    "COMMUNICATION_ROUNDS": 5,
    "NUM_CLUSTERS": 3
}

class Server():
    def __init__(self, server_id, train_dataloader_iid, train_dataloader_non_iid, num_clients, is_iid=True, num_rounds=model_constants["COMMUNICATION_ROUNDS"], clients_ids=[], client_fraction=model_constants["CLIENT_FRACTION"]):
        self.server_id = server_id
        self.client_ids = clients_ids
        self.clients_per_round: int = math.ceil(client_fraction * len(self.client_ids))
        self.batch_size = model_constants["LOCAL_MINIBATCH"]

        self.server_cv: threading.Condition = threading.Condition(threading.Lock())

        # list: index is client id and value is tuple of (weights, loss)
        self.clients_training_data: list[tuple] = [(0,0)] * num_clients
        self.devices_done_running: set = set()

        self.global_model: MNISTCNN = MNISTCNN(model_constants)

        self.total_rounds = num_rounds
        self.num_rounds = num_rounds

        if is_iid:
            train_dataloader=train_dataloader_iid
        else:
            train_dataloader=train_dataloader_non_iid

        print("Done splitting the data.")
        self.clients = OrderedDict()
        for client_id in self.client_ids:
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


    def start(self, load_from_checkpoint=False):
        # Main training loop
        self.server_cv.acquire()
        checkpoint_dir = pathlib.Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        # breakpoint()
        if load_from_checkpoint:
            # find last checkpoint round (round with best loss)
            try:
                matches = glob.glob(str(checkpoint_dir / f"server{self.server_id:05d}*"))
                best_checkpoint = sorted(matches)[-1]
                self.global_model.load_state_dict(torch.load(best_checkpoint, weights_only=True))
                self.kill_clients()
                return None
            except (IndexError, FileNotFoundError) as e:
                print(f"Checkpoint not loaded successfully for server {self.server_id}:", e)
        best_loss = np.inf
        while not self.convergence_criteria():
            print("\n\n########################################################################\n", end="")
            print(f"SERVER INITIALIZING COMMUNICATION ROUND #{self.total_rounds - self.num_rounds}\n", end="")
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
            accuracy, loss = self.test_model()
            if loss < best_loss:
                # checkpoint current model if loss has improved
                best_loss = loss
                checkpoint_path = checkpoint_dir / f"server{self.server_id:05d}-round{(self.total_rounds - self.num_rounds):05d}"
                torch.save(self.global_model.state_dict(), checkpoint_path)
                

        self.server_cv.release()

        # Signal all devices to exit and return when they are done
        self.kill_clients()
        
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
        all_predictions = []
        all_labels = []
        total_loss = 0.0

        # Define loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Disable gradient calculation for testing (increases efficiency)
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move inputs and labels to the appropriate device (GPU or CPU)
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
                
                # Get model predictions
                outputs = self.global_model(inputs)

                # calculate batch loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # get model prediction
                _, predictions = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy and loss
        avg_loss = total_loss / len(test_loader)
        accuracy = sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Loss: {avg_loss:.2f}')
        return (accuracy, avg_loss)


    def kill_clients(self):
        for client in self.clients.values():
            client.kill()
            client.join()