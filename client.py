from enum import Enum

from torch.utils.data import DataLoader
from models import MNISTCNN
import threading

class DeviceAction(Enum):
    RUN: int = 0
    WAIT: int = 1
    STOP: int = 2

class Client():
    def __init__(self, client_id: int, server, train_dataloader: DataLoader):
        self.client_id = client_id
        self.server = server

        self.client_cv = threading.Condition(threading.Lock())
        self.server_cv = server.get_server_cv()

        self.status = DeviceAction.WAIT

        self.thread = threading.Thread(target=self.start)

        self.model: MNISTCNN = None

        self.train_dataloader = train_dataloader

        self.thread.start()


    def start(self):
        # print(f"Client {self.client_id} started.\n", end="")
        self.client_cv.acquire()
        while self.status != DeviceAction.STOP:

            while self.status == DeviceAction.WAIT:
                # print(f"Client {self.client_id} waiting for directions.\n", end="")
                self.client_cv.wait()
            
            if self.status == DeviceAction.STOP:
                self.client_cv.release()
                return
            
            # Let server know that the client is done running
            # print(f"Client {self.client_id} is training.\n", end="")
            self.server.send_client_result(self.client_id, self.model.train_model(self.train_dataloader, self.client_id))

            self.status = DeviceAction.WAIT
            with self.server_cv:
                self.server_cv.notify()

        self.client_cv.release()


    def kill(self):
        with self.client_cv:
            self.status = DeviceAction.STOP
            self.client_cv.notify()
        return


    def run_training(self):
        with self.client_cv:
            self.status = DeviceAction.RUN
            self.client_cv.notify()

        return


    def send_global_model(self, model: MNISTCNN):
        self.model = model


    def join(self):
        self.thread.join()
