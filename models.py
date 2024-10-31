import torch
from torch import nn
import torch.nn.functional as F

from main import model_constants

class MNISTCNN(nn.Module):
    def __init__(self):
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
    

    def train(self, client_id):
        dataloader = []
        average_loss = -1
        
        # TODO: handle batch sizes, dependent on dataloader structure - Abhi?
        for _ in range(self.local_epochs):
            total_loss = 0
            for inputs, labels in dataloader[client_id]['train']:
                inputs, labels = input.to('cpu'), labels.to('cpu')

                self.optimizer.zero_grad()
                predictions = self(inputs)

                curr_loss = self.loss_function(predictions, labels)
                curr_loss.backward()

                self.optimizer.step()

                total_loss += curr_loss.item()

            average_loss = total_loss / len(dataloader[client_id]['train'])
        

        # We can access a model's weights with model.state_dict()
        # We also need to save the loss to plot it
        assert(average_loss != -1)
        return self.state_dict(), average_loss