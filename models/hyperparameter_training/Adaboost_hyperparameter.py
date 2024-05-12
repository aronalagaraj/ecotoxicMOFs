# Importing necessary packages
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

path = 'Ada_params'
if not os.path.exists(path=path):  # Create a new directory if it doesn't exist
    os.makedirs(path)

# Converting the inputs and outputs to numpy arrays
x = np.array(descriptors)
X = np.nan_to_num(x.astype(np.float32))
y = np.array(class_data['2-Class'])

# Creating the hyperparameter space with hyperopt
space = {'base_depth': hp.quniform('base_depth', 1, 10, 1),  # Depth of initial tree
         'n_estimators': hp.quniform('n_estimators', 10, 1000, 10),  # Maximum number of estimators at which boosting
         # is terminated
         'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),  # Weight applied to each
         # classifier at each boosting iteration
         'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R'])  # Choice of boosting algorithm
         }


def hyperparameter_tuning(space):
    # n_jobs=-1 uses all cores. tree_method refers to how is the tree is constructed, 'hist' is the fastest
    base = DecisionTreeClassifier(max_depth=int(space['base_depth']))
    ada = AdaBoostClassifier(random_state=42, estimator=base, n_estimators=int(space['n_estimators']),
                             learning_rate=space['learning_rate'], algorithm=space['algorithm'])

    acc_mean_train, rec_mean_train, acc_test, rec_test = scores(ada, X_train, y_train, X_test, y_test)

    # Using recall as the loss
    loss = 1 - rec_test

    return {'loss': loss, 'status': STATUS_OK, 'model': ada}


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

    # Defining these as the output for best_params is a number for the choice
    algorithm = ['SAMME', 'SAMME.R']

    # Using the optimised parameters to define a test model
    base = DecisionTreeClassifier(max_depth=int(best_params['base_depth']))
    test = AdaBoostClassifier(random_state=42, estimator=base, n_estimators=int(best_params['n_estimators']),
                              learning_rate=best_params['learning_rate'], algorithm=algorithm[best_params['algorithm']])
    
    acc_mean_train, rec_mean_train, acc_test, rec_test = scores(test, X_train, y_train, X_test, y_test)

    train_acc[j] = acc_mean_train
    test_acc[j] = acc_test

    # Store into a file
    dump(best_params, f'{path}/best_params_{j}.joblib')

print(f'Highest test accuracy was {np.max(test_acc)} with {np.argmax(test_acc) * 10} features.')

print('Completed')
print(f'--- {time.time() - start_time} seconds ---')
