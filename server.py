from client import Client
import threading


class Server():
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.clients = [Client(i, self) for i in range(num_clients)]

        self.client_cvs = [threading.Condition(threading.Lock()) for i in range(num_clients)]
        self.server_cv = threading.Condition(threading.Lock())

        self.clients_training_data = [] * num_clients
        self.devices_done_running = set()


    def get_client_cv(self, client_id: int):
        return self.client_cvs[client_id]


    def get_server_cv(self):
        return self.server_cv


    def send_client_result(self, device_id: int, client_result: tuple):
        self.clients_training_data[device_id] = client_result
        self.devices_done_running.add(device_id)