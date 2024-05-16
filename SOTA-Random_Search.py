# compare SOTA algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
from joblib import dump, load


if __name__ == "__main__":
    
    # load data
    training_set = pd.read_csv('./data/Training_set.csv')
    x_train = np.array(training_set.iloc[:,:-1])
    y_train = np.array(training_set.iloc[:,-1:]).reshape(-1)
    x_train, y_train = shuffle(x_train, y_train, random_state=1)
    
    
    # NB
    NB = RandomizedSearchCV(cv=5, estimator=GaussianNB(), n_iter=25, n_jobs = -1, param_distributions={'var_smoothing': [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]}, scoring='accuracy', verbose=5)
    
    search = NB.fit(x_train, y_train)
    print(search.best_params_)
    estimator = NB.best_estimator_
    dump(estimator, "./model/NB_best.joblib")
    
    
    # RFSMMA
    RFSMMA = RandomizedSearchCV(cv=5, estimator=RandomForestClassifier(), n_iter=25, n_jobs=-1, param_distributions={'n_estimators': [25, 50, 100, 150, 200], 'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': [None, 10, 20, 30], 'max_features': ['sqrt', 'log2', None], 'min_samples_split': [2,3,4]}, scoring='accuracy', verbose=5)
    
    search = RFSMMA.fit(x_train, y_train)
    print(search.best_params_)
    estimator = RFSMMA.best_estimator_
    dump(estimator, "./model/RFSMMA_best.joblib")
    
    
    # XGBoost
    XGB = RandomizedSearchCV(cv=5, estimator=XGBClassifier(), n_iter=25, n_jobs=-1, param_distributions={'booster': ['gbtree', 'gblinear'], 'colsample_bytree': [0.8, 0.9, 1], 'gamma': [1, 2, 3], 'learning_rate': [0.1, 0.01, 0.001], 'max_depth': [7, 10, 15, 20], 'min_child_weight': [10, 15, 20, 25], 'n_estimators': [50, 100, 200, 300, 400, 500, 600], 'objective': ['binary:logistic'],'verbosity':[0]}, scoring='accuracy', verbose=5)
    
    search = XGB.fit(x_train, y_train)
    print(search.best_params_)
    estimator = XGB.best_estimator_
    dump(estimator, "./model/XGB_best.joblib")
    
    