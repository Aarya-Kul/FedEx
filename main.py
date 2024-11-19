from server import Server
import numpy as np
import pdb
from sklearn.cluster import SpectralClustering
from collections import OrderedDict
from mnist_dataloader import MNISTDataloader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import argparse
import pathlib
import pickle
import torch
from models import FedEx

np.random.seed(42)
torch.manual_seed(42)
# Main thread logic for assigning tasks
def main():
    parser = argparse.ArgumentParser(description="program that runs FedAvg. Optionally with clustering or weight logs")
    parser.add_argument(
        "num_clients", type=int, help="Number of clients"
    )
    parser.add_argument(
        "-c", "--clusters", type=int, default=-1, help="Number of clusters (optional). If not used, no clustering is run."
    )

    # Parse the arguments
    args = parser.parse_args()

    num_clients = int(args.num_clients)
    # pass dataloaders (iid and non-iid) split in to servers.
    transform = transforms.Compose([transforms.ToTensor()])
    # 60,000 pictures 
    train_mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 10,000 pictures
    # test_mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # create dataloaders for each function
    examples_per_client = len(train_mnist_data) // num_clients
    shard_size = examples_per_client // 2 # two shards per client
    
    train_dataloader_iid = MNISTDataloader(dataset=train_mnist_data, 
                                                num_clients = num_clients,
                                                shard_size=shard_size,
                                                is_iid=True,
                                                val_ratio=0)
    train_dataloader_non_iid = MNISTDataloader(dataset=train_mnist_data, 
                                                num_clients = num_clients,
                                                shard_size=shard_size,
                                                is_iid=False,
                                                val_ratio=0)

    # RUN CLUSTERING
    #########
    if int(args.clusters) != -1:
        print("Running clustering")
        num_clusters = int(args.clusters)
        # NUM_CLUSTERS = 3
        print(f"Load clusters from checkpoint? [y/n]")
        print(">> ", end="")
        load_clusters = input()
        if load_clusters in ["y", "yes", "Yes"]:
            load_clusters = True
        else:
            load_clusters = False
        clusters = run_clustering(num_clusters, num_clients, train_dataloader_iid, train_dataloader_non_iid, load_from_checkpoint = load_clusters)
        cluster_servers = OrderedDict()
        for server_id, cluster_devices in enumerate(clusters):
            cluster = Server(
                server_id=server_id,
                is_iid=False,
                train_dataloader_iid=train_dataloader_iid,
                train_dataloader_non_iid=train_dataloader_non_iid,
                num_clients=num_clients,
                clients_ids=cluster_devices
            )
            print(f"Load server model {server_id} from checkpoint? [y/n]")
            print(">> ", end="")
            load_from_checkpoint = input()
            if load_from_checkpoint in ["y", "yes", "Yes"]:
                load_from_checkpoint = True
            else:
                load_from_checkpoint = False
            cluster.start(load_from_checkpoint=load_from_checkpoint)
            cluster_servers[server_id] = cluster

        callFedEx(cluster_servers)
    else:
        print("Running without clustering")
        figure_2_params = [[10, 1], [10, 5], [10,20], [50, 1], [50, 5],
                           [50, 20], [float('inf'), 1], 
                           [float('inf'), 5], [float('inf'), 20]]
        
        server1 = Server(
            server_id=0,
            is_iid=False,
            train_dataloader_iid=train_dataloader_iid,
            train_dataloader_non_iid=train_dataloader_non_iid,
            num_clients=num_clients, 
            num_rounds = 10, 
            clients_ids=range(num_clients),
            local_epochs=10,
            batch_size=10
        )
        server2 = Server(
            server_id=0,
            is_iid=False,
            train_dataloader_iid=train_dataloader_iid,
            train_dataloader_non_iid=train_dataloader_non_iid,
            num_clients=num_clients, 
            num_rounds = 10, 
            clients_ids=range(num_clients),
            local_epochs=10,
            batch_size=5
        )
            
        server1.start() 
        server2.start()
        test_accuracies = [server1.test_accuracies, server2.test_accuracies]
        labels = ["B=10 E=10", "B=5 E=10"]
        gen_plot(test_accuracies, labels)

    # At this point, all the clients are done running
    # TODO: modify the server to cache all the losses to reproduce graphs from paper
    print("Training complete")
    # TODO: modify server to save weights to file so that they can be used for inference

def run_clustering(num_clusters, num_clients, train_dataloader_iid, train_dataloader_non_iid, load_from_checkpoint=False):
        # load from checkpoint if requested
        checkpoint_dir = pathlib.Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_filename = checkpoint_dir / "clustering.pkl"
        if load_from_checkpoint:
            # Load the clusters from checkpoint
            print("Loading clusters from checkpoint...")
            with open(checkpoint_filename, 'rb') as file:
                clusters = pickle.load(file)
            print(f"cluster assignments: {clusters}")
            return clusters
            
        # step 1: run current core model and keep track of weights
        initial_server = Server(server_id=-1, clients_ids=range(num_clients), num_rounds=1, client_fraction=1, num_clients=num_clients, train_dataloader_iid=train_dataloader_iid, train_dataloader_non_iid=train_dataloader_non_iid)
        clients_training_data = initial_server.start(load_from_checkpoint=False)


        # step 2: cluster weights to find similar clients
        stored_weights = []
        for client_weights, _ in clients_training_data:
            #stored_weights.append(np.concatenate([weights.flatten() for weights in client_weights.values()]))
            curr_row = []
            for weights in client_weights.values():
                curr_row.extend(weights.flatten().tolist())
            stored_weights.append(curr_row)

        # feature_matrix = np.concatenate(**clients_training_data.values()[0])

        feature_matrix = np.array(stored_weights)

        # calculate cosine similarity
        similarity_matrix = cosine_similarity(feature_matrix)
        print(similarity_matrix)
        
        clustering = SpectralClustering(n_clusters = num_clusters, assign_labels='discretize', affinity='precomputed', random_state=0).fit(similarity_matrix)
        clusters = [np.where(clustering.labels_ == i)[0] for i in range(num_clusters)]
        with open(checkpoint_filename, 'wb') as file:
            pickle.dump(clusters, file)
        return clusters


def callFedEx(cluster_servers):
    # train_dataloader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])))
    
    models = [server.global_model for server in cluster_servers.values()]
    fedex = FedEx(models=models)
    fedex.test_model()

def gen_plot(client_test_data, labels):
    plt.figure(figsize=(8, 6))
    communication_rounds = np.linspace(1, 10, 10) 
    for i in range(len(client_test_data)): 
        plt.plot(communication_rounds, client_test_data[i], 'r-', label=labels[i])
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("MNIST CNN IID", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure_2_replication.png")
    plt.show()


if __name__ == '__main__':
    main()