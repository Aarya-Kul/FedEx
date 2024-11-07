from enum import Enum
# from server import Server
from models import MNISTCNN
import threading

class DeviceAction(Enum):
    RUN: int = 0
    WAIT: int = 1
    STOP: int = 2

class Client():
    def __init__(self, client_id: int, server):
        self.client_id = client_id
        self.server = server

        self.client_cv = threading.Condition(threading.Lock())
        self.server_cv = server.get_server_cv()

        self.status = DeviceAction.WAIT

        self.thread = threading.Thread(target=self.start())

        self.model: MNISTCNN = MNISTCNN()


    def start(self):
        self.client_cv.acquire()
        while self.status != DeviceAction.STOP:

            while self.status == DeviceAction.WAIT:
                self.client_cv.wait()

            # Let server know that the client is done running
            self.server.send_client_result(self.client_id, self.model.train(self.client_id))

            self.status = DeviceAction.WAIT
            self.server_cv.notify()

        self.client_cv.release()


    def kill(self):
        self.status = DeviceAction.STOP
        self.client_cv.notify()
        return


    def run_training(self):
        self.status = DeviceAction.RUN
        self.client_cv.notify()
        return


    def send_global_model(self, model: MNISTCNN):
        self.model = model


    def join(self):
        self.thread.join()
