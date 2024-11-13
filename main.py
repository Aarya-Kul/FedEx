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
import argparse
import pathlib
import pickle


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
                                                is_iid=True)
    train_dataloader_non_iid = MNISTDataloader(dataset=train_mnist_data, 
                                                num_clients = num_clients,
                                                shard_size=shard_size,
                                                is_iid=False)


    # RUN CLUSTERING
    #########
    if int(args.clusters) != -1:
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
    else:
        server = Server(
            server_id=0,
            train_dataloader_iid=train_dataloader_iid,
            train_dataloader_non_iid=train_dataloader_non_iid,
            num_clients=num_clients,
            clients_ids=range(num_clients)
        )
        server.start()
    

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


if __name__ == '__main__':
    main()