from client import Client
from models import MNISTCNN
import numpy as np
import threading
from collections import OrderedDict
import torch
import copy

from main import model_constants

class Server():
    def __init__(self, num_rounds=1000):
        self.num_clients: int = model_constants["NUM_DEVICES"]
        self.clients: list[Client] = [Client(i, self) for i in range(self.num_clients)]

        self.server_cv: threading.Condition = threading.Condition(threading.Lock())

        self.clients_training_data: list[int] = [] * self.num_clients
        self.devices_done_running: set = set()

        self.global_model: MNISTCNN = MNISTCNN()

        self.num_rounds = num_rounds


    def get_server_cv(self):
        return self.server_cv


    def send_client_result(self, device_id: int, client_result: tuple):
        self.clients_training_data[device_id] = client_result
        self.devices_done_running.add(device_id)
    
    def convergence_criteria(self):
        # Note: could experiment with stopping when the model reaches a certain accuracy
        self.num_rounds -= 1
        if self.num_rounds <= 0:
            return True
        return False


    def start(self):
        # Main training loop
        self.server_cv.acquire()
        while not self.convergence_criteria():
            devices_to_run = np.random.choice(range(self.num_clients), size=self.num_clients, replace=False)
            self.devices_done_running.clear()

            # Signal devices to run
            for device in devices_to_run:
                self.clients[device].send_global_model(copy.deepcopy(self.global_model))
                self.clients[device].run_training()

            # Wait for devices to be done running
            while len(self.devices_done_running) != len(devices_to_run):
                self.server_cv.wait()

            # Update global model to reflect client updates
            self.global_model.load_state_dict(self.fed_avg())

        self.server_cv.release()

        # Signal all devices to exit and return when they are done
        for i in range(self.num_clients):
            self.clients[i].kill()
            self.clients[i].join()


    # Return the state dictionary of the global model after training
    def fed_avg(self, weight_log = False):
        DUMMY_WEIGHTS = (np.random.rand(self.num_clients) + 1) * 42 # list of data sizes for each client (>1)
        
        # take log of weights if testing extension 2
        if weight_log:
            DUMMY_WEIGHTS = np.log(DUMMY_WEIGHTS)

        first_weight = self.devices_training_data[0][0]
        
        avg_weights = OrderedDict()
        for key in first_weight.keys():
            curr_weights = [DUMMY_WEIGHTS[i] * np.array(state[key]) for i, (state, _) in enumerate(self.devices_training_data)]
            stacked_weights = torch.stack(curr_weights)
            avg_weights[key] = torch.mean(stacked_weights)

        return avg_weights


    # EXTENSION 2: 
    def fed_avg_log(self):
        self.fed_avg(weight_log=True)