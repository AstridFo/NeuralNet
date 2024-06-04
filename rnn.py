# References
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# https://github.com/hunkim/PyTorchZeroToAll/blob/master/09_2_softmax_mnist.py
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
from torch import cuda
import torchvision
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import ast
import torch.nn.functional as F



def create_data():
    # Define the range for random numbers
    num_range = range(1, 10)  # Example range from 1 to 9

    # Generate random rows
    num_rows = 100  # specify the number of rows you want
    row1 = [random.choice(num_range) for _ in range(num_rows)]
    row2 = [random.choice(num_range) for _ in range(num_rows)]
    row3 = [random.choice(num_range) for _ in range(num_rows)]
    row4 = [random.choice(num_range) for _ in range(num_rows)]
    row5 = [random.choice(num_range) for _ in range(num_rows)]

    animal_list = ["['hest']", "['ku']", "['gris']", "['sau']"]
    parsed_a = [ast.literal_eval(item)[0] for item in animal_list]
    random_animals = [[random.choice(parsed_a)] for _ in range(num_rows)]
    # print(random_animals)

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()
    # Create the DataFrame
    df = pd.DataFrame()
    df = pd.DataFrame(random_animals, columns=['farm_animals'])
    # Fit and transform the 'farm_animals' column
    df['farm_animals_encoded'] = label_encoder.fit_transform(df['farm_animals'])

    # print(df)
    # Create the DataFrame
    df1 = pd.DataFrame({
        'row1': row1,
        'row2': row2,
        'row3': row3,
        'row4': row4,
        'row5': row5
    })


    # Assume the target labels are the sum of the rows modulo 2 (just for demonstration)
    df1['target'] = df['farm_animals_encoded']
    print(df1)
    return df1



#
device = 'cuda' if cuda.is_available() else 'cpu'
#device = 'cpu'

Epoch_num = 20
batch_size = 100

train_loss_values = np.zeros([Epoch_num - 1])
test_loss_values = np.zeros([Epoch_num - 1])


data = create_data()

class Train_Dataset(Dataset):
    """  dataset."""
    def __init__(self, data):
        xy = np.array(data)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, :-1]).float()
        self.y_data = from_numpy(xy[:, [-1]]).long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Test_Dataset(Dataset):
    """  dataset."""
    def __init__(self, data):
        xy = np.array(data)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, :-1]).float()
        self.y_data = from_numpy(xy[:, [-1]]).long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

train_dataset = Train_Dataset(data)
test_dataset = Test_Dataset(data)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(5, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.l3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.l4 = nn.Linear(32, 7)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.l1(x)))
        # x = self.dropout(x)
        x = F.relu(self.bn2(self.l2(x)))
        # x = self.dropout(x)
        x = F.relu(self.bn3(self.l3(x)))
        # x = self.dropout(x)
        x = self.l4(x)
        return F.softmax(x, dim=1)

# our model
model = Model()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

def train(epoch, train_loss_values):
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(inputs)
        labels = torch.reshape(labels, [-1]).to(device)
        loss = criterion(y_pred, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} | Batch Status: {batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%) | Loss: {loss.item():.6f}')
    train_loss /= len(train_loader.dataset)
    train_loss_values[epoch-1] = train_loss
    return train_loss_values

def test(test_loss_values):
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_pred = model(inputs)
            labels = torch.reshape(labels, [-1]).to(device)
            loss = criterion(y_pred, labels).item()
            test_loss += loss
            pred = y_pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_loss_values[epoch-1] = test_loss
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')
    return test_loss_values, all_preds, all_labels

if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, Epoch_num):
        epoch_start = time.time()
        train_loss_values = train(epoch, train_loss_values)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test_loss_values, all_preds, all_labels = test(test_loss_values)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
    print(train_loss_values)
    print(test_loss_values)

    # Print predictions vs true labels
    for i in range(len(all_preds)):
        print(f'Prediction: {all_preds[i]}, True Label: {all_labels[i]}')

    # Plotting loss
    x_len = np.arange(len(train_loss_values))
    plt.plot(x_len, test_loss_values, marker='.', label="Test-set Loss")
    plt.plot(x_len, train_loss_values, marker='.', label="Train-set Loss")
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
