import threading
import time
from enum import Enum
import numpy as np
import models
import copy
import torch
from collections import OrderedDict
from server import Server



# Define constants for federated learning
model_constants = {
    "NUM_DEVICES": 16,
    "DEVICES_PER_EPOCH": 4,
    "LOCAL_MINIBATCH": 10,
    "LOCAL_EPOCHS": 1000,
    "LEARNING_RATE": 0.1,
    "EXAMPLES_PER_CLIENT": 3750,
    "LABELS_PER_CLIENT": 2
}

class DeviceAction(Enum):
    RUN: int = 0
    WAIT: int = 1
    STOP: int = 2


# Main thread logic for assigning tasks
def main():
    server = Server()
    server.start()

    # At this point, all the clients are done running
    # TODO: modify the server to cache all the losses to reproduce graphs from paper
    print("Training complete")
    # TODO: modify server to save weights to file so that they can be used for inference



if __name__ == '__main__':
    main()
