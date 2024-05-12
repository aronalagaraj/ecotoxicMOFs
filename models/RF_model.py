# Importing necessary packages
from sklearn.ensemble import RandomForestClassifier
from model_training import *

# Manually fine-tuned parameters after coarse hyperparameter tuning
params = {'bootstrap': True, 'criterion': 'gini', 'max_depth': 30, 'max_features': None,
          'min_samples_leaf': 4, 'min_samples_split': 7, 'n_estimators': 1000}
rfc = RandomForestClassifier(**params, random_state=0, verbose=0, n_jobs=-1)

threshold = 0.4
run_model(rfc, threshold)
