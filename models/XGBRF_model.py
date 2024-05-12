# Importing necessary packages
from xgboost import XGBRFClassifier
from model_training import *

# Manually fine-tuned parameters after coarse hyperparameter tuning
params = {'tree_method': 'approx', 'alpha': 0.055, 'eta': 0.3, 'gamma': 0.092, 'max_depth': 20,
          'min_child_weight': 0.23, 'n_estimators': 56, 'reg_lambda': 0.16, 'objective': 'binary:logistic',
          'subsample': 0.8, 'colsample_bytree': 0.95, 'colsample_bylevel': 0.90, 'colsample_bynode': 0.25}
xgbrf = XGBRFClassifier(**params, random_state=0, verbose=0, n_jobs=-1)

threshold = 0.4
run_model(xgbrf, threshold)
