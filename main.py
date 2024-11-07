# import threading
# import time
from enum import Enum
# import numpy as np
# import models
# import copy
# import torch
# from collections import OrderedDict
from server import Server

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
