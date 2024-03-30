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


training_set = pd.read_csv('./data/Training_set.csv')
x_train = np.array(training_set.iloc[:,:-1])
y_train = np.array(training_set.iloc[:,-1:]).reshape(-1)
x_train, y_train = shuffle(x_train, y_train, random_state=1)
    
    
def SMTRI():
    SMTRI = tf.keras.models.load_model('./model/best_DNN_model.h5')
    return SMTRI


def NB():
    NB = GaussianNB(var_smoothing=1e-8).fit(x_train, y_train)
    return NB


def RFSMMA():
    RFSMMA = RandomForestClassifier(n_estimators= 25, min_samples_split= 4, max_features= 'log2', max_depth= 10, criterion= 'log_loss').fit(x_train, y_train)
    return RFSMMA


def XGB():
    XGB = XGBClassifier(objective= 'binary:logistic', n_estimators= 200, min_child_weight= 20, max_depth= 20, learning_rate= 0.01, gamma= 2, colsample_bytree= 0.8, booster= 'gbtree').fit(x_train, y_train)
    return XGB



if __name__ == "__main__":
    