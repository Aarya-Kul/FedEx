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

# Enforce ordering constraints for devices
device_locks = [threading.Lock()] * NUM_DEVICES
device_cvs = [threading.Condition(lock=device_locks[i]) for i in range(NUM_DEVICES)]

# Allow manager to sleep while clients are running
manager_lock = threading.Lock()
manager_cv = threading.Condition(lock=manager_lock)

# Devices add themselves to this set when they are done running
devices_done_running = set()

# Function to represent work done by worker threads
def device(device_id):
    device_cvs[device_id].acquire()
    while device_signals[device_id] != DeviceAction.STOP:

        while device_signals[device_id] == DeviceAction.WAIT:
            device_cvs[device_id].wait()

        print("TODO: implement device model")

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

        # Signal devices to run
        for device in devices_to_run:
            device_signals[device] = DeviceAction.RUN
            device_cvs[device].notify()

        # Wait for devices to be done running
        while len(devices_done_running) != len(devices_to_run):
            manager_cv.wait()
        
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
    manager_cv.release()


# TODO: Define
def convergence_criteria():
    return False

if __name__ == '__main__':
    main()
