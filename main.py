import threading
import time
from enum import Enum
import numpy as np
import models
import copy
import torch
from collections import OrderedDict



# Define constants for federated learning
NUM_DEVICES = 16
DEVICES_PER_EPOCH = 4
LOCAL_MINIBATCH = 10
LOCAL_EPOCHS = 1000
LEARNING_RATE = 0.1
EXAMPLES_PER_CLIENT = 3750
LABELS_PER_CLIENT = 2

class DeviceAction(Enum):
    RUN: int = 0
    WAIT: int = 1
    STOP: int = 2


# Signals so that devices know when to run
device_signals = [DeviceAction.WAIT] * NUM_DEVICES

# List to hold the device weights and loss device_id -> (device_weight, device_loss)
devices_training_data = [] * NUM_DEVICES

# Enforce ordering constraints for devices
device_locks = [threading.Lock() for _ in range(NUM_DEVICES)]
device_cvs = [threading.Condition(lock=device_locks[i]) for i in range(NUM_DEVICES)]

# Allow manager to sleep while clients are running
manager_lock = threading.Lock()
manager_cv = threading.Condition(lock=manager_lock)

# Devices add themselves to this set when they are done running
devices_done_running = set()

# Global server model
global_model = models.MNISTCNN()
optimizer = torch.optim.Adam(global_model.parameters, lr=LEARNING_RATE)
loss_function = torch.nn.CrossEntropyLoss()

# Function to represent work done by worker threads
def device(device_id):
    device_cvs[device_id].acquire()
    while device_signals[device_id] != DeviceAction.STOP:

        while device_signals[device_id] == DeviceAction.WAIT:
            device_cvs[device_id].wait()

        devices_training_data[device_id]= train(device_id)

        devices_done_running.add(device_id)
        device_signals[device_id] = DeviceAction.WAIT
        manager_cv.notify()

    device_cvs[device_id].release()


# Main thread logic for assigning tasks
def main():
    manager_cv.acquire()
    # Create and start worker threads
    workers = []
    for i in range(NUM_DEVICES):
        thread = threading.Thread(target=device, args=(i, device_signals))
        thread.start()
        workers.append(thread)

    # Main training loop
    while not convergence_criteria():
        devices_to_run = np.random.choice(range(NUM_DEVICES), size=DEVICES_PER_EPOCH, replace=False)
        devices_done_running.clear()

        # Signal devices to run
        for device in devices_to_run:
            device_signals[device] = DeviceAction.RUN
            device_cvs[device].notify()

        # Wait for devices to be done running
        while len(devices_done_running) != len(devices_to_run):
            manager_cv.wait()

        # Update global model to reflect client updates
        global_model.load_state_dict(fed_avg())

        # TODO: handle graphs
    
    # Signal all devices to exit
    for i in range(NUM_DEVICES):
        device_signals[i] = DeviceAction.STOP
    
    # Wait for all workers to finish
    for thread in workers:
        thread.join()

    # TODO: save model weights to file for future use
    print("Training complete.")
    manager_cv.release()


# TODO: Define
def convergence_criteria():
    return False


# Client trainning loop
def train(device_id):
    local_model = copy.deepcopy(global_model)
    dataloader = []
    average_loss = -1
    
    # TODO: handle batch sizes, dependent on dataloader structure - Abhi?
    for _ in range(LOCAL_EPOCHS):
        total_loss = 0
        for inputs, labels in dataloader[device_id]['train']:
            inputs, labels = input.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()
            predictions = local_model(inputs)

            curr_loss = loss_function(predictions, labels)
            curr_loss.backward()

            optimizer.step()

            total_loss += curr_loss.item()

        average_loss = total_loss / len(dataloader[device_id]['train'])
    

    # We can access a model's weights with model.state_dict()
    # We also need to save the loss to plot it
    assert(average_loss != -1)
    return local_model.state_dict(), average_loss


# Return the state dictionary of the global model after training
def fed_avg(weight_log = False):
    DUMMY_WEIGHTS = (np.random.rand(NUM_DEVICES) + 1) * 42 # list of data sizes for each client (>1)
    
    # take log of weights if testing extension 2
    if weight_log:
        DUMMY_WEIGHTS = np.log(DUMMY_WEIGHTS)

    first_weight = devices_training_data[0][0]
    
    avg_weights = OrderedDict()
    for key in first_weight.keys():
        curr_weights = [DUMMY_WEIGHTS[i] * np.array(state[key]) for i, (state, _) in enumerate(devices_training_data)]
        stacked_weights = torch.stack(curr_weights)
        avg_weights[key] = torch.mean(stacked_weights)

    return avg_weights


# EXTENSION 2: 
def fed_avg_log():
    fed_avg(weight_log=True)
    

if __name__ == '__main__':
    main()
