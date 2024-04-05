import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from joblib import dump, load
import os
import time

start_time = time.time()

device = torch.device("mps") if torch.backends.mps.is_available() else None

print('Device:', device)

complete_data = pd.read_csv('Data/complete_data.csv')
descriptors = pd.read_csv('Data/descriptors.csv')
class_data = pd.read_csv('Data/class_data.csv')

x = np.array(descriptors)
X = np.nan_to_num(x.astype(np.float32))

y = np.array(class_data['2-Class'])
y = (y + 1) / 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

space = {
    'lr': hp.loguniform('lr', np.log(0.00001), np.log(0.01)),
    'n_nodes': hp.quniform('n_nodes', 60, 600, 20),
    'epochs': hp.quniform('epochs', 500, 2000, 10),
    'pos_weight': hp.uniform('pos_weight', 1.5, 3)
}

torch.manual_seed(42)


class ANN(nn.Module):
    def __init__(self, n_nodes):
        super(ANN, self).__init__()

        self.input = nn.Linear(199, n_nodes)
        self.act0 = nn.ReLU()

        self.hidden1 = nn.Linear(n_nodes, n_nodes)
        self.act1 = nn.ReLU()

        self.hidden2 = nn.Linear(n_nodes, n_nodes)
        self.act2 = nn.ReLU()

        self.hidden3 = nn.Linear(n_nodes, n_nodes)
        self.act3 = nn.ReLU()

        self.output = nn.Linear(n_nodes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act0(self.input(x))
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.sigmoid(self.output(x))
        return x


def hyperparameter_tuning(space):
    model = ANN(n_nodes=int(space['n_nodes'])).to(device)
    pos_weight = torch.ones([1], device=device) * space['pos_weight']
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.SGD(model.parameters(), lr=space['lr'])

    for epoch in range(int(space['epochs'])):
        model.train()

        # Forward Pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train.view(-1, 1))

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch: {epoch}  Loss: {loss}')
    print('\n')

    model.eval()
    with torch.no_grad():
        y_eval = model.forward(X_test)
        test_loss = criterion(y_eval, y_test.view(-1, 1))

    return {'loss': test_loss.item(), 'status': STATUS_OK}


trials = Trials()
best = fmin(fn=hyperparameter_tuning, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

print('Best parameters:', best)
dump(best, f'best_model_ann_loop.joblib')
#
# best = load('best_model_ann_loop.joblib')
#
model = ANN(n_layers=int(best['n_layers']), n_nodes=int(best['n_nodes'])).to(device)
pos_weight = torch.ones([1], device=device) * best['pos_weight']
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=best['lr'])

for epoch in range(int(best['epochs'])):
    model.train()

    # Forward Pass
    y_pred = model(X_train, n_layers=int(best['n_layers']))
    loss = criterion(y_pred, y_train.view(-1, 1))

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == int(best['epochs']-1):
        print(f'Epoch: {epoch}  Loss: {loss}')

model.eval()
with torch.no_grad():
    y_pred = model(X_train, n_layers=int(best['n_layers']))
    y_eval = model(X_test, n_layers=int(best['n_layers']))
    train_loss = criterion(y_pred, y_train.view(-1, 1))
    test_loss = criterion(y_eval, y_test.view(-1, 1))

print(train_loss)
print(test_loss)


def acc_score(model, X, y, space):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(X):
            y_val = model(data)
            y_val = y_val.round()

            if y_val.item() == y[i]:
                correct += 1
            total += 1

    return correct / total


def rec_score(model, X, y, space):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for i, data in enumerate(X):
            y_val = model(data)
            y_val = y_val.round()

            if y_val.item() == y[i] and y[i] == 1:
                tp += 1
            elif y_val.item() == y[i] and y[i] == 0:
                tn += 1
            elif y_val.item() != y[i] and y[i] == 1:
                fp += 1
            elif y_val.item() != y[i] and y[i] == 0:
                fn += 1

    return tp / (tp + fn)


print(f'Train Accuracy: {acc_score(model, X_train, y_train, best)}')
print(f'Test Accuracy: {acc_score(model, X_test, y_test, best)}')

print(f'Train Recall: {rec_score(model, X_train, y_train, best)}')
print(f'Test Recall: {rec_score(model, X_test, y_test, best)}')

print("Completed")
print("--- %s seconds ---" % (time.time() - start_time))
