# Importing necessary packages
from sklearn.metrics import accuracy_score, recall_score, make_scorer
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold


def scores(model, X_train, y_train, X_test, y_test):
    # Using accuracy and recall as the metrics
    scoring = {'accuracy': make_scorer(accuracy_score),
               'recall': make_scorer(recall_score)}

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    cv.get_n_splits(X_train, y_train)

    # Cross-validation on recall and accuracy
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

    acc_mean_train = scores['test_accuracy'].mean()
    acc_std_train = scores['test_accuracy'].std()
    print(f"{acc_mean_train:.2f} accuracy with a standard deviation of {acc_std_train:.2f}")

    rec_mean_train = scores['test_recall'].mean()
    rec_std_train = scores['test_recall'].std()
    print(f"{rec_mean_train:.2f} recall with a standard deviation of {rec_std_train:.2f}")

    # Calculating the scores for the test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred)
    rec_test = recall_score(y_test, y_pred)

    print(f'Test accuracy: {acc_test:.2f}')
    print(f'Test recall: {rec_test:.2f}')

    return acc_mean_train, rec_mean_train, acc_test, rec_test

