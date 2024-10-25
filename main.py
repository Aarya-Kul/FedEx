import threading
import time
from enum import Enum
import numpy as np


# Define the number of device threads
NUM_DEVICES = 16
DEVICES_PER_EPOCH = 4

class DeviceAction(Enum):
    RUN: int = 0
    WAIT: int = 1
    STOP: int = 2


# Signals so that devices know when to run
device_signals = [DeviceAction.WAIT] * NUM_DEVICES

# Devices add themselves to this set when they are done running
devices_done_running = set()

# Function to represent work done by worker threads
def device(device_id, task_queue):
    while device_signals[device_id] == DeviceAction.WAIT:

        if device_signals[device_id] == DeviceAction.RUN:
            print("TODO: implement device model")
            # Run epoch on device model
            # TODO: implement device model

        # Avoid busy waiting
        time.sleep(0.1)


# Main thread logic for assigning tasks
def main():
    # Create and start worker threads
    workers = []
    for i in range(NUM_DEVICES):
        thread = threading.Thread(target=device, args=(i, device_signals))
        thread.start()
        workers.append(thread)
    
    # Main training loop
    while not convergence_criteria():
        devices_to_run = np.random.choice(range(NUM_DEVICES), size=DEVICES_PER_EPOCH, replace=False)

        # Signal devices to run
        for device in devices_to_run:
            device_signals[device] = DeviceAction.RUN
        
        # Wait for devices to be done running
        while len(devices_done_running) != len(devices_to_run):
            time.sleep(0.1)
        
        # Aggregate model results
        # TODO: define aggregation method
    
    # Signal all devices to exit
    for i in range(NUM_DEVICES):
        device_signals[i] = DeviceAction.STOP
    
    # Wait for all workers to finish
    for thread in workers:
        thread.join()

    # TODO: save model weights to file for future use
    print("Training complete.")


# TODO: Define
def convergence_criteria():
    return False

main()
