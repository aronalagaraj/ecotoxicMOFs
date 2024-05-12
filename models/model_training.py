# Importing necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer, matthews_corrcoef, balanced_accuracy_score, accuracy_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scoring(model, X, y):
    # Defining the scoring functions. For more context as to why these metrics are chosen, or how they are
    # calculated, please refer to the dissertation where some of these concepts are discussed

    scoring = {'accuracy': make_scorer(accuracy_score),
               'balanced_accuracy': make_scorer(balanced_accuracy_score),
               'f1_macros': 'f1_macro',
               'MCC': make_scorer(matthews_corrcoef)}

    # Cross-validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
    cv.get_n_splits(X, y)

    print('Starting cross-validation')
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    print('Finished cross-validation')

    acc_mean = scores['test_accuracy'].mean()
    acc_std = scores['test_accuracy'].std()
    print(f"{acc_mean:.2f} accuracy with a standard deviation of {acc_std:.2f}")

    bacc_mean = scores['test_balanced_accuracy'].mean()
    bacc_std = scores['test_balanced_accuracy'].std()
    print(f"{bacc_mean:.2f} balanced accuracy with a standard deviation of {bacc_std:.2f}")

    f1_mean = scores['test_f1_macros'].mean()
    f1_std = scores['test_f1_macros'].std()
    print(f"{f1_mean:.2f} f1-macro with a standard deviation of {f1_std:.2f}")

    mcc_mean = scores['test_MCC'].mean()
    mcc_std = scores['test_MCC'].std()
    print(f"{mcc_mean:.2f} MCC with a standard deviation of {mcc_std:.2f}")

    print('Fitting model')
    model.fit(X, y)

    return model, scores


# Loading the data, need to specify the file path
# It's recommended that the data is stored in an identical format to the training data that we have provided.
def data():
    # Change the path as needed
    pathX = 'data/training_data/descriptors.csv'
    pathy = 'data/training_data/class_data.csv'

    X = pd.read_csv(pathX)
    y = pd.read_csv(pathy)
    y = y['2-Class']

    print('Data loaded successfully')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4000)
    print('Data split successfully')

    # Normalising data, fit and transform separately train and validation data separately
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    print('Scaling done successfully')

    # Selecting k features using the default f_classif score function
    k = 100
    selector = SelectKBest(f_classif, k=k)
    X_train = selector.fit_transform(X_train, y_train)
    X_val = selector.transform(X_val)
    print('Feature selection done successfully')

    return X_train, X_val, y_train, y_val, scaler, selector


def plot_roc_curve(y, y_pred):
    roc_auc = roc_auc_score(y, y_pred)  # AUCROC calculation requires probabilities
    fpr, tpr, thresholds = roc_curve(y, y_pred)  # Curve generation

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')  # Adding the AUCROC in the legend
    ax.plot([0, 1], [0, 1], label='Random Forest', linestyle='--', color='black')
    ax.legend(loc='lower right')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.01])
    ax.set_xticks(np.linspace(0, 1.0, 11))
    ax.set_yticks(np.linspace(0, 1.0, 11))
    plt.show()


def plot_confusion_matrix(y, y_pred_prob, threshold):
    # Generation of the confusion matrix
    y_pred = (y_pred_prob >= threshold).astype('int')
    ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap=plt.cm.Blues, display_labels=['Non-Toxic', 'Toxic'])
    plt.show()


def prob_score(y, y_pred_prob, threshold):
    # Calculation of the accuracy and recall scores
    y_pred = (y_pred_prob >= threshold).astype('int')
    return accuracy_score(y, y_pred), recall_score(y, y_pred)


def test(scaler, selector):
    # Loading the test data
    X_test = pd.read_csv('data/unseen_data/unseen_descriptors.csv')
    y_test = pd.read_csv('data/unseen_data/unseen_data.csv')

    X_test = scaler.fit_transform(X_test)
    X_test = selector.transform(X_test)
    y_test = y_test['2-Class']

    return X_test, y_test


# Running the model
def run_model(model, threshold):
    X_train, X_val, y_train, y_val, scaler, selector = data()
    model, _ = scoring(model, X_train, y_train)

    # Metrics and plots for the training data
    y_train_pred = model.predict_proba(X_train)[:, 1]
    plot_roc_curve(y_train, y_train_pred)
    plot_confusion_matrix(y_train, y_train_pred, threshold)
    train_acc, train_rec = prob_score(y_train, y_train_pred, threshold)
    print(f'Train Accuracy score: {train_acc:.2f}')
    print(f'Train Recall score: {train_rec:.2f}')

    # Metrics and plots for the validation data
    y_val_pred = model.predict_proba(X_val)[:, 1]
    plot_roc_curve(y_val, y_val_pred)
    plot_confusion_matrix(y_val, y_val_pred, threshold)
    val_acc, val_rec = prob_score(y_val, y_val_pred, threshold)
    print(f'Validation Accuracy score: {val_acc:.2f}')
    print(f'Validation Recall score: {val_rec:.2f}')

    # Metrics and plots for the test data
    X_test, y_test = test(scaler, selector)
    y_test_pred = model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_test_pred)
    plot_confusion_matrix(y_test, y_test_pred, threshold)
    test_acc, test_rec = prob_score(y_test, y_test_pred, threshold)
    print(f'Test Accuracy score: {test_acc:.2f}')
    print(f'Test Recall score: {test_rec:.2f}')

    return model
