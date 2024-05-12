# Importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay

# Importing PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryRecall

import time

start_time = time.time()

# This was done on a Macbook with Apple Silicon chip, which can use its GPU for training known as MPS
device = torch.device("mps") if torch.backends.mps.is_available() else None

# If NVIDIA GPU is available, uncomment the following line to use it
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"

print('Device:', device)

# Loading the data
complete_data = pd.read_csv('data/training_data/complete_data.csv')
descriptors = pd.read_csv('data/training_data/descriptors.csv')
class_data = pd.read_csv('data/training_data/class_data.csv')

X = np.array(descriptors)
y = np.array(class_data['2-Class'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_test = np.array(pd.read_csv('data/unseen_data/unseen_descriptors.csv'))
y_test = pd.read_csv('Data/unseen_data.csv')
y_test = np.array(y_test['2-Class'])

# Normalising data, fit and transform separately train and validation data separately
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

# Converting data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

torch.manual_seed(42)
threshold = 0.4


# Defining the Artificial Neural Network
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        # Arbitrary number of neurons with an arbitrary number of layers
        n_hidden_1 = 500
        n_hidden_2 = 500
        n_hidden_3 = 500

        self.input = nn.Linear(199, n_hidden_1)
        self.act0 = nn.ReLU()  # Activation function, ReLU chosen

        self.hidden1 = nn.Linear(n_hidden_1, n_hidden_2)
        self.act1 = nn.ReLU()

        self.hidden2 = nn.Linear(n_hidden_2, n_hidden_3)
        self.act2 = nn.ReLU()

        self.output = nn.Linear(n_hidden_3, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid function chosen for binary classification

    def forward(self, x):  # Forward pass
        x = self.act0(self.input(x))
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.sigmoid(self.output(x))
        return x


# Defining the DataLoaders with a batch size of 30
batch_size = 30

y_train = y_train.unsqueeze(1)  # Adding a dimension to the tensor
train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  # Creating an iterable over the dataset

y_val = y_val.unsqueeze(1)
val_ds = TensorDataset(X_val, y_val)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

y_test = y_test.unsqueeze(1)
test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

model = ANN().to(device)  # Creating instance of class in device
learning_rate = 0.001  # Learning rate is the step size each iteration
criterion = nn.BCEWithLogitsLoss()  # Loss function chosen, Binary Cross Entropy with Logits for binary classification
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Optimiser for updating weights
# (Stochastic Gradient Descent)

# Number of iterations and lists to track the loss and accuracy
epochs = 2001
train_loss = []
train_acc = []
val_loss = []
val_acc = []

for epoch in range(epochs):
    # Training the model
    model.train()
    correct_train = 0
    total_train = 0
    temp_rec_train = []
    # Iterating over the DataLoader
    for X_b, y_b in train_dl:
        # Forward Pass
        y_pred = model(X_b)
        loss = criterion(y_pred, y_b)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating the accuracy
        threshold_pred = (y_pred > threshold).int()
        correct_train += (threshold_pred == y_b).sum().item()
        total_train += y_b.size(0)

        recall = BinaryRecall(threshold=threshold).to(device=device)
        temp_rec_train.append(recall(y_pred.view(-1), y_b.view(-1)).item())

    # Validating the model
    model.eval()
    correct_val = 0
    total_val = 0
    temp_rec_val = []
    with torch.no_grad():
        for X_b, y_b in val_dl:
            y_pred = model(X_b)
            loss_val = criterion(y_pred, y_b)

            threshold_pred = (y_pred > threshold).int()
            correct_val += (threshold_pred == y_b).sum().item()
            total_val += y_b.size(0)

            recall = BinaryRecall(threshold=threshold).to(device=device)
            temp_rec_val.append(recall(y_pred.view(-1), y_b.view(-1)).item())

    if epoch % 50 == 0:
        train_loss.append(loss.item())
        val_loss.append(loss_val.item())
        train_acc.append(correct_train / total_train)
        val_acc.append(correct_val / total_val)
        print(
            f'Epoch: [{epoch} / {epochs - 1}],  Train Loss: {loss.item():.2f},  Train Acc: {correct_train / total_train:.2f}, Train Rec: {mean(temp_rec_train):.2f}, '
            f'Val Loss: {loss_val.item():.2f}, Val Acc: {correct_val / total_val:.2f}, Val Rec: {mean(temp_rec_val):.2f}')

# Plotting the loss and accuracy over the epochs
iter = range(0, epochs, 50)

plt.plot(iter, train_loss)
plt.plot(iter, val_loss)
plt.show()

plt.plot(iter, train_acc)
plt.plot(iter, val_acc)
plt.show()

# Testing the model
model.eval()
with torch.no_grad():
    correct_test = 0
    total_test = 0
    temp_rec_test = []
    for X_b, y_b in test_dl:
        y_pred = model(X_b)
        loss_test = criterion(y_pred, y_b)

        threshold_pred = (y_pred > threshold).int()
        correct_test += (threshold_pred == y_b).sum().item()
        total_test += y_b.size(0)

        recall = BinaryRecall(threshold=threshold).to(device=device)
        temp_rec_test.append(recall(y_pred.view(-1), y_b.view(-1)).item())

print(
    f'Test Loss: {loss_test.item():.2f}, Test Acc: {correct_test / total_test:.2f}, Test Rec: {mean(temp_rec_test):.2f}')
print(list(model.parameters())[0])


# Functions to plot ROC curve and confusion matrix
def roc_auc(model, loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_b, y_b in loader:
            y_eval = model(X_b)
            y_pred.extend(y_eval.cpu().detach().numpy())  # When converting back to numpy array, detach from GPU to CPU
            y_true.extend(y_b.cpu().detach().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plot_roc_curve(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)


def plot_roc_curve(y, y_pred):
    roc_auc = roc_auc_score(y, y_pred)
    fpr, tpr, thresholds = roc_curve(y, y_pred)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], label='Random', linestyle='--', color='black')
    ax.legend(loc='lower right')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.01])
    ax.set_xticks(np.linspace(0, 1.0, 11))
    ax.set_yticks(np.linspace(0, 1.0, 11))
    plt.show()


def plot_confusion_matrix(y, y_pred_prob):
    y_pred = (y_pred_prob >= threshold).astype('int')
    ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap=plt.cm.Blues, display_labels=['Non-Toxic', 'Toxic'])
    plt.show()


roc_auc(model, train_dl)
roc_auc(model, val_dl)
roc_auc(model, test_dl)

print("Completed")
print("--- %s seconds ---" % (time.time() - start_time))
