# Importing necessary packages
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRFClassifier
from hyperopt import fmin, tpe, STATUS_OK, Trials, hp

from joblib import dump, load
import os
import time

from scores import scores

start_time = time.time()

# Importing data, change path as needed
data_path = 'training_data'
complete_data = pd.read_csv(f'{data_path}/complete_data.csv')
descriptors = pd.read_csv(f'{data_path}/descriptors.csv')
class_data = pd.read_csv(f'{data_path}/class_data.csv')

path = 'XGBRF_params'
if not os.path.exists(path=path):  # Create a new directory if it doesn't exist
    os.makedirs(path)

# Converting the inputs and outputs to numpy arrays
x = np.array(descriptors)
X = np.nan_to_num(x.astype(np.float32))
y = np.array(class_data['2-Class'])

# Creating the hyperparameter space with hyperopt
space = {'n_estimators': hp.quniform('n_estimators', 10, 1000, 10),  # Number of trees in forest
         'max_depth': hp.quniform("max_depth", 15, 35, 1),  # Maximum depth of each tree
         'eta': hp.loguniform('eta', np.log(0.0001), np.log(0.1)),
         # Learning rate, step size shrinkage to prevent overfitting
         'gamma': hp.uniform('gamma', 0, 2),
         # Minimum loss reduction required  to make a further partition on a leaf node of the tree
         'alpha': hp.uniform('alpha', 0, 2),  # L1 regularization term
         'reg_lambda': hp.uniform('reg_lambda', 0, 3),  # L2 regularization term
         'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),  # Subsample ratio of columns when constructing
         # each tree
         'min_child_weight': hp.uniform('min_child_weight', 0, 1)  # Minimum sum of instance weight needed in a child
         }


def hyperparameter_tuning(space):
    # n_jobs=-1 uses all cores. tree_method refers to how is the tree is constructed, 'hist' is the fastest
    xgbrf = XGBRFClassifier(tree_method='hist', random_state=42, n_jobs=-1, n_estimators=int(space['n_estimators']),
                            max_depth=int(space['max_depth']), eta=space['eta'], gamma=space['gamma'],
                            alpha=space['alpha'], reg_lambda=space['reg_lambda'],
                            colsample_bytree=space['colsample_bytree'],
                            min_child_weight=space['min_child_weight'])

    acc_mean_train, rec_mean_train, acc_test, rec_test = scores(xgbrf, X_train, y_train, X_test, y_test)

    # Using recall as the loss
    loss = 1 - rec_test

    return {'loss': loss, 'status': STATUS_OK, 'model': xgbrf}


#  To store the accuracies for the features checked
train_acc = np.zeros(20)
test_acc = np.zeros(20)

for j in range(20):
    # For continued use, in case of the file being interrupted
    if os.path.isfile(f'{path}/best_params_{j}.joblib'):
        print(f"{path}/best_params_{j}.joblib, exists, skipping to next iteration.")
        continue

    # Testing every 10 features from 10 up to 199
    k = (j + 1) * 10 if j != 19 else 199

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalising data, fit and transform separately train and test data separately
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Selecting k features using the default f_classif score function
    selector = SelectKBest(f_classif, k=k)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # The actual hyperopt hyperparameter tuning, Trials() enables a progress bar to visualise the progress
    trials = Trials()
    best_params = fmin(hyperparameter_tuning, space, algo=tpe.suggest, max_evals=50, trials=trials)
    print(f'The best parameters are: {best_params}')

    # Using the optimised parameters to define a test model
    test = XGBRFClassifier(tree_method='hist', random_state=42, n_jobs=-1, n_estimators=int(best_params['n_estimators']),
                           max_depth=int(best_params['max_depth']), eta=best_params['eta'], gamma=best_params['gamma'],
                           alpha=best_params['alpha'], reg_lambda=best_params['reg_lambda'],
                           colsample_bytree=best_params['colsample_bytree'], min_child_weight=best_params['min_child_weight'])

    acc_mean_train, rec_mean_train, acc_test, rec_test = scores(test, X_train, y_train, X_test, y_test)

    train_acc[j] = acc_mean_train
    test_acc[j] = acc_test

    # Store into a file
    dump(best_params, f'{path}/best_params_{j}.joblib')

print(f'Highest test accuracy was {np.max(test_acc)} with {np.argmax(test_acc) * 10} features.')

print('Completed')
print(f'--- {time.time() - start_time} seconds ---')
