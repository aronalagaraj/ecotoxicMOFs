# Importing necessary packages
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, STATUS_OK, Trials, hp

from joblib import dump, load
import os
import time

from scores import scores

# One method to speed up the training of SVMs is to use the scikit-learn-intelex package, however, requires an Intel
# CPU therefore not available to all
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Successfully imported scikit-learn-intelex and patched sklearn.")
except ImportError:
    print("scikit-learn-intelex is not installed.")


start_time = time.time()

# Importing data, change path as needed
data_path = 'training_data'
complete_data = pd.read_csv(f'{data_path}/complete_data.csv')
descriptors = pd.read_csv(f'{data_path}/descriptors.csv')
class_data = pd.read_csv(f'{data_path}/class_data.csv')

path = 'SVM_params'
if not os.path.exists(path=path):  # Create a new directory if it doesn't exist
    os.makedirs(path)

# Converting the inputs and outputs to numpy arrays
x = np.array(descriptors)
X = np.nan_to_num(x.astype(np.float32))
y = np.array(class_data['2-Class'])

# Creating the hyperparameter space with hyperopt
space = {'C': hp.lognormal('C', 0, 1),  # Regularisation parameter
         'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf']),  # Type of kernel used: 'linear' = linear kernel,
         # 'poly' = polynomial kernel and 'rbf' = radial basis function kernel
         'degree': hp.choice('degree', [2, 3, 4]),  # The degree for the polynomial kernel, ignored for other kernels
         'gamma': hp.uniform('gamma', 0, 20),  # Kernel coefficient for 'rbf' or 'poly'
         'shrinking': hp.choice('shrinking', [True, False])  # Whether to use shrinking heuristic
         }


def hyperparameter_tuning(space):
    # n_jobs is unavailable for scikit's SVM
    svm = SVC(random_state=42, C=space['C'], kernel=space['kernel'],
              gamma=space['gamma'], degree=space['degree'], shrinking=space['shrinking'])

    acc_mean_train, rec_mean_train, acc_test, rec_test = scores(svm, X_train, y_train, X_test, y_test)

    # Using recall as the loss
    loss = 1 - rec_test

    return {'loss': loss, 'status': STATUS_OK, 'model': svm}


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
    kernel = ['linear', 'poly', 'rbf']
    degree = [2, 3, 4]

    # Using the optimised parameters to define a test model
    test = SVC(random_state=42, C=best_params['C'], kernel=kernel[best_params['kernel']], gamma=best_params['gamma'],
               degree=degree[best_params['degree']], shrinking=best_params['shrinking'])


    acc_mean_train, rec_mean_train, acc_test, rec_test = scores(test, X_train, y_train, X_test, y_test)

    train_acc[j] = acc_mean_train
    test_acc[j] = acc_test

    # Store into a file
    dump(best_params, f'{path}/best_params_{j}.joblib')

print(f'Highest test accuracy was {np.max(test_acc)} with {np.argmax(test_acc) * 10} features.')

print('Completed')
print(f'--- {time.time() - start_time} seconds ---')
