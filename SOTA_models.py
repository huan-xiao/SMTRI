# compare SOTA algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier


# set network seed to make it reproducible
keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()


x_train = np.load('./data/x_train.npy')
y_train = np.load('./data/y_train.npy')
    
    
def SMTRI():
    SMTRI = tf.keras.models.load_model('./model/best_DNN_model.h5')
    return SMTRI


def NB():
    NB = GaussianNB().fit(x_train, y_train)
    return NB


def RFSMMA():
    RFSMMA = RandomForestClassifier(n_estimators=100, max_features=0.2, min_samples_leaf=10).fit(x_train, y_train)
    return RFSMMA


def SNMFSMMA():
    SNMFSMMA = RidgeClassifier(alpha=0.5).fit(x_train, y_train)
    return SNMFSMMA


def SMAJL():
    SMAJL = LogisticRegression(C=0.01, solver = 'newton-cg').fit(x_train, y_train)
    return SMAJL


def RNAmigos():
    RNAmigos = MLPClassifier(hidden_layer_sizes=100, tol=0.1).fit(x_train, y_train)
    return RNAmigos


def XGB():
    XGB = XGBClassifier(n_estimators=100, max_depth=15, objective='binary:logistic').fit(x_train, y_train)
    return XGB

def KNN():
    KNN = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    return KNN


def SVM():
    SVM = SVC(probability=True, kernel = 'linear', C=0.1).fit(x_train, y_train)
    return SVM


if __name__ == "__main__":
    RNAmigos()