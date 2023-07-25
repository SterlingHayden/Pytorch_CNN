#                   ----------------------  CNN version of MLP_Pytorch_NN.ipynb and NN_Pythorch.Script.py  ----------------------

import numpy as np
import pickle 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


#############################################################################################
### Working with data
data_dict = pickle.load(open("cifar-2class-py2/cifar_2class_py2.p", "rb"), encoding="bytes")
train_data = data_dict[b'train_data']
train_labels = data_dict[b'train_labels']
test_data = data_dict[b'test_data']
test_labels = data_dict[b'test_labels']
m, n = train_data.shape # n = 32 x 32 pixels X 3 for RGB

# Create Function to create our
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

# Convert data and labels to torch tensors
train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels[:, 0]).long()  # Use only the first column as labels
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels[:, 0]).long()  # Use only the first column as labels

# Create train and test datasets
train = CustomDataset(train_data, train_labels)
test = CustomDataset(test_data, test_labels)

# Create Dataloaders
batch_size = 10  # Define batch size
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch_size)

# Calculate tensor height and width
# batch = next(iter(train_dataloader))[0]
# (height, width) = batch.shape

#############################################################################################
# Define the Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(5,5))
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=20, kernel_size=(4,4))  # number in_chanels = previous out_channels
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7056,100),  # I have no clue why its 7056 other than the linear algebra matrix multiply rules
            nn.ReLU(),
            nn.Linear(100,2)#,
            #nn.Softmax()s
        )

    def forward(self, X):
        # X = self.pool(nn.functional.relu(self.conv1(X)))
        # X = nn.functional.relu(self.conv2(X))
        X = nn.functional.relu(self.conv1(X))
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        prob = F.softmax(logits, dim=1)
        return prob
    

model = NeuralNetwork().to(device)
print(model)


#############################################################################################
### Param Tuning
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0, momentum=1e-4)


#############################################################################################
### Create Train Function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.reshape(batch_size, 3, 32, 32) #added so we can process RGB in model

        optimizer.zero_grad()
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


#############################################################################################
### Create Test function
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X,y) in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.reshape(batch_size, 3, 32, 32) #added so we can process RGB in model
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#############################################################################################
### Training and Evaluation Loop
epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")