import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from hyperopt import fmin, tpe, STATUS_OK, Trials, hp, space_eval

from xgboost import XGBRFClassifier
from imblearn.over_sampling import RandomOverSampler

from joblib import dump, load
import os
import time
start_time = time.time()

complete_data = pd.read_csv('Data/complete_data.csv')
descriptors = pd.read_csv('Data/descriptors.csv')
class_data = pd.read_csv('Data/class_data.csv')

path = 'Imbalance_loop_recall_4'
if not os.path.exists(path=path):
    os.makedirs(path)

x = np.array(descriptors)
X = np.nan_to_num(x.astype(np.float32))

y = np.array(class_data['2-Class'])
y = (y + 1) / 2

oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_over, y_over = oversample.fit_resample(X, y)

space = {'n_estimators': hp.quniform('n_estimators', 1000, 2000,1),
         'max_depth': hp.quniform("max_depth", 15, 35, 1),
         'eta': hp.loguniform('eta', np.log(0.0001), np.log(0.1)),
         'gamma': hp.uniform('gamma', 0, 2),
         'alpha': hp.uniform('alpha', 0, 2),
         'reg_lambda': hp.uniform('reg_lambda', 0, 3),
         'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
         'min_child_weight': hp.uniform('min_child_weight', 0, 1)
         }


def hyperparameter_tuning(space):
    xgbrf = XGBRFClassifier(tree_method='hist', random_state=42, n_jobs=-1, n_estimators=int(space['n_estimators']),
                            max_depth=int(space['max_depth']),  eta=space['eta'], gamma=space['gamma'],
                            alpha=space['alpha'], reg_lambda=space['reg_lambda'], colsample_bytree=space['colsample_bytree'],
                            min_child_weight=space['min_child_weight'])

    xgbrf.fit(X_train, y_train)

    # recall_train = recall_score(y_train, xgbrf.predict(X_train))
    # recall_test = recall_score(y_test, xgbrf.predict(X_test))

    recall_train = cross_val_score(xgbrf, X_train, y_train, cv=StratifiedKFold(10), scoring='recall')
    recall_test = cross_val_score(xgbrf, X_test, y_test, cv=StratifiedKFold(10), scoring='recall')
    delta = abs(accuracy_score(y_train, xgbrf.predict(X_train)) - accuracy_score(y_test, xgbrf.predict(X_test)))

    print(f"Train Recall: {np.mean(recall_train)}")
    print(f"Test Recall: {np.mean(recall_test)}")
    print(f'Train Accuracy: {accuracy_score(y_train, xgbrf.predict(X_train))}')
    print(f'Test Accuracy: {accuracy_score(y_test, xgbrf.predict(X_test))}')
    print(f"Delta: {delta} \n")

    loss = 1 - np.mean(recall_test)

    return {'loss': loss, 'status': STATUS_OK, 'model': xgbrf}


train_acc = np.zeros(20)
test_acc = np.zeros(20)
deltas = np.zeros(20)

for j in range(20):
    k = (j + 1) * 10 if j != 19 else 199

    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42, stratify=y_over)

    selector = SelectKBest(k=k)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    trials = Trials()
    best_i = fmin(hyperparameter_tuning, space, algo=tpe.suggest, max_evals=200, trials=trials)
    print(best_i)

    test = XGBRFClassifier(tree_method='hist', random_state=42, n_jobs=-1, n_estimators=int(best_i['n_estimators']),
                            max_depth=int(best_i['max_depth']),  eta=best_i['eta'], gamma=best_i['gamma'],
                            alpha=best_i['alpha'], reg_lambda=best_i['reg_lambda'], colsample_bytree=best_i['colsample_bytree'],
                            min_child_weight=best_i['min_child_weight'])

    test.fit(X_train, y_train)

    recall_train = recall_score(y_train, test.predict(X_train))
    recall_test = recall_score(y_test, test.predict(X_test))
    delta = abs(accuracy_score(y_train, test.predict(X_train)) - accuracy_score(y_test, test.predict(X_test)))

    print(f"Train set Recall: {recall_train}")
    print(f"Test set Recall: {recall_test}")

    print(f"Train set Accuracy: {accuracy_score(y_train, test.predict(X_train))}")
    print(f"Test set Accuracy: {accuracy_score(y_test, test.predict(X_test))}")

    train_acc[j] = recall_train
    test_acc[j] = recall_test
    deltas[j] = delta

    dump(best_i, f'{path}/best_model_recall_loop_{j}.joblib')
    dump(test, f'{path}/test_model_recall_loop_{j}.joblib')


print(f'Highest test accuracy was {np.max(test_acc)} at i = {np.argmax(test_acc)}, with a delta of {deltas[np.argmax(test_acc)]}.')
print(f'Lowest delta was {np.min(deltas)} at i = {np.argmin(deltas)}, where train accuracy = {train_acc[np.argmin(deltas)]} and test accuracy = {test_acc[np.argmin(deltas)]}.')
print('\n')


print("Completed")
print("--- %s seconds ---" % (time.time() - start_time))