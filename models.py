import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class MNISTCNN(nn.Module):
    def __init__(self, model_constants):
        super(MNISTCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

        # The paper is not really specific about how the fully connected layers are built
        # I think that kind of makes sense
        # 28 -> 24 -> 12 -> 8 -> 4
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10) 

        self.optimizer = torch.optim.Adam(self.parameters(), lr=model_constants["LEARNING_RATE"])
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.local_epochs = model_constants["LOCAL_EPOCHS"]


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        output = F.softmax(x, dim=1)

        return output
    

    def train_model(self, train_dataloader: DataLoader, client_id: int):
        print("Training model within model class")
        average_loss = -1

        # Detect device
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {device}")

        # Move model to device
        self.to(device)
        self.train()
        for epoch in range(self.local_epochs):                
            total_loss = 0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                predictions = self(inputs)

                curr_loss = self.loss_function(predictions, labels)
                curr_loss.backward()

                self.optimizer.step()

                total_loss += curr_loss.item()

            average_loss = total_loss / len(train_dataloader)
            if epoch % 2 == 0:
                print(f'Client {client_id}: Epoch [{epoch}/{self.local_epochs}], Loss: {average_loss:.4f}\n', end="")

        # We can access a model's weights with model.state_dict()
        # We also need to save the loss to plot it
        assert(average_loss != -1)
        return self.state_dict(), average_loss


    # def test(self, test_dataloader: DataLoader):
    #     average_loss = -1
        
    #     for _ in range(self.local_epochs):
    #         total_loss = 0
    #         for inputs, labels in test_dataloader:
    #             inputs, labels = inputs.to('cpu'), labels.to('cpu')

    #             self.optimizer.zero_grad()
    #             predictions = self(inputs)

    #             curr_loss = self.loss_function(predictions, labels)
    #             curr_loss.backward()

    #             self.optimizer.step()

    #             total_loss += curr_loss.item()

    #         average_loss = total_loss / len(train_dataloader)


    #     # We can access a model's weights with model.state_dict()
    #     # We also need to save the loss to plot it
    #     assert(average_loss != -1)
    #     return self.state_dict(), average_loss


class FedEx(nn.Module):
    def __init__(self, num_clusters, learning_rate, epochs, models, train_dataloader):
        print("FedEx order recieved (initializing)\n", end="")
        self.fc1 = nn.Linear(in_features=num_clusters * 10, out_features=num_clusters * 10 * 4)
        self.fc2 = nn.Linear(in_features=num_clusters * 10 * 4, out_features=num_clusters * 10 * 2) 
        self.fc3 = nn.Linear(in_features=num_clusters * 10 * 2, out_features=num_clusters * 10)
        self.fc4 = nn.Linear(in_features=num_clusters * 10, out_features=10)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

        slelf.learning_rate = learning_rate
        self.epochs = epochs

        self.models = models
        self.train_dataloader = train_dataloader


    def forward(self, x):
        print("Your FedEx package is in transit (forward pass)\n", end="")
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2)

        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.2)

        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.2)

        output = F.softmax(x, dim=1)
        return output
    

    def train_model(self):
        print("Your FedEx package just left the warehouse (training...)\n", end="")
        average_loss = -1

        # Detect device
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # Move model to device
        self.to(device)
        self.train()
        for epoch in range(self.epochs):  
            # Inference on all the clusters

            total_loss = 0
            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                fedEx_inputs = []
                for cnn in self.models:
                    cnn.eval()
                    fedEx_inputs += list(cnn(inputs))
                
                fedEx_inputs = torch.tensor(fedEx_inputs)

                self.optimizer.zero_grad()
                predictions = self(fedEx_inputs)

                curr_loss = self.loss_function(predictions, labels)
                curr_loss.backward()

                self.optimizer.step()

                total_loss += curr_loss.item()

            average_loss = total_loss / len(self.train_dataloader)
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.epochs}], Loss: {average_loss:.4f}\n', end="")

        print("Your FedEx package was succesfully delivered (training done)\n", end="")

        # We can access a model's weights with model.state_dict()
        # We also need to save the loss to plot it
        assert(average_loss != -1)
        return self.state_dict(), average_loss
    
    def test_model(self):
        test_mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = DataLoader(test_mnist_data)
        # Set the model to evaluation mode
        self.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0

        # Define loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Disable gradient calculation for testing (increases efficiency)
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move inputs and labels to the appropriate device (GPU or CPU)
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
                
                # Get model predictions
                outputs = self(inputs)

                # calculate batch loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # get model prediction
                _, predictions = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy and loss
        avg_loss = total_loss / len(test_loader)
        accuracy = sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Loss: {avg_loss:.2f}')
        return (accuracy, avg_loss)