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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from joblib import dump, load
from main import *
import warnings
warnings.filterwarnings("ignore", message="WARNING")


keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
np.random.seed(0)

    
def get_metrics(y_test, y_pred):
    
    predictions = np.round(y_pred)
    
    AUC = round(roc_auc_score(y_test, y_pred),3)
    Accuracy = round(accuracy_score(y_test, predictions),3)
    Precision = round(precision_score(y_test, predictions),3)
    Recall = round(recall_score(y_test, predictions),3)
    F1_Score = round(f1_score(y_test, predictions),3)
    MCC = round(matthews_corrcoef(y_test, predictions),3)
    Kappa = round(cohen_kappa_score(y_test, predictions),3)
    
    return [AUC, Accuracy, Precision, Recall, F1_Score, MCC, Kappa]
        
    
def cross_validation(model, model_name, X, y):
    kf = KFold(n_splits=5)
    AUC=[]
    Accuracy=[]
    Precision=[]
    Recall=[]
    F1_Score=[]
    MCC=[]
    Kappa=[]
    
    tprs = []
    base_fpr = np.linspace(0, 1, 100)
    
    
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        this_model = model
        
        if "SMTRI" in model_name:
            this_model.fit(X_train, y_train, epochs=78, batch_size=32, verbose=0)
            y_pred = this_model.predict(X_test, verbose=0)
        else:
            this_model.fit(X_train, y_train)
            y_pred = this_model.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        
        metrics = get_metrics(y_test, y_pred)
        AUC.append(metrics[0])
        Accuracy.append(metrics[1])
        Precision.append(metrics[2])
        Recall.append(metrics[3])
        F1_Score.append(metrics[4])
        MCC.append(metrics[5])
        Kappa.append(metrics[6])
        
        
    print("\n***** "+model_name+" *****")
    print('AUC: '+f"{round(np.mean(AUC),3)}"+u"\u00B1"+f"{round(np.std(AUC),3)}")
    print('Accuracy: '+f"{round(np.mean(Accuracy),3)}"+u"\u00B1"+f"{round(np.std(Accuracy),3)}")
    print('Precision: '+f"{round(np.mean(Precision),3)}"+u"\u00B1"+f"{round(np.std(Precision),3)}")
    print('Recall: '+f"{round(np.mean(Recall),3)}"+u"\u00B1"+f"{round(np.std(Recall),3)}")
    print('F1-Score: '+f"{round(np.mean(F1_Score),3)}"+u"\u00B1"+f"{round(np.std(F1_Score),3)}")
    print('MCC: '+f"{round(np.mean(MCC),3)}"+u"\u00B1"+f"{round(np.std(MCC),3)}")
    print('Kappa: '+f"{round(np.mean(Kappa),3)}"+u"\u00B1"+f"{round(np.std(Kappa),3)}")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    
    return mean_tpr

    
if __name__ == "__main__":
    
    # load three testing datasets
    testing_set_1 = pd.read_csv('./data/PDB_testing_set.csv')
    x_1 = np.array(testing_set_1.iloc[:,:-1])
    y_1 = np.array(testing_set_1.iloc[:,-1:]).reshape(-1)
    x_1, y_1 = shuffle(x_1, y_1, random_state=1)
    
    testing_set_2 = pd.read_csv('./data/PubChem_testing_set.csv')
    x_2 = np.array(testing_set_2.iloc[:,:-1])
    y_2 = np.array(testing_set_2.iloc[:,-1:]).reshape(-1)
    x_2, y_2 = shuffle(x_2, y_2, random_state=1)
    
    testing_set_3 = pd.read_csv('./data/RPocket_testing_set.csv')
    x_3 = np.array(testing_set_3.iloc[:,:-1])
    y_3 = np.array(testing_set_3.iloc[:,-1:]).reshape(-1)
    x_3, y_3 = shuffle(x_3, y_3, random_state=1)
    
    
    # SMTRI
    SMTRI = tf.keras.models.load_model('./model/DNN_model.h5')
    
    SMTRI_PDB_tpr = cross_validation(SMTRI, "SMTRI_PDB", x_1, y_1)
    SMTRI_PubChem_tpr = cross_validation(SMTRI, "SMTRI_PubChem", x_2, y_2)
    SMTRI_RPocket_tpr = cross_validation(SMTRI, "SMTRI_RPocket", x_3, y_3)
    
    
    # XGBoost
    XGB = load('./model/XGB_best.joblib')
    
    XGB_PDB_tpr = cross_validation(XGB, "XGB_PDB", x_1, y_1)
    XGB_PubChem_tpr = cross_validation(XGB, "XGB_PubChem", x_2, y_2)
    XGB_RPocket_tpr = cross_validation(XGB, "XGB_RPocket", x_3, y_3)
    
    
    # NB
    NB = load('./model/NB_best.joblib')
    
    NB_PDB_tpr = cross_validation(NB, "NB_PDB", x_1, y_1)
    NB_PubChem_tpr = cross_validation(NB, "NB_PubChem", x_2, y_2)
    NB_RPocket_tpr = cross_validation(NB, "NB_RPocket", x_3, y_3)
    
    
    # RFSMMA
    RFSMMA = load('./model/RFSMMA_best.joblib')
    
    RFSMMA_PDB_tpr = cross_validation(RFSMMA, "RFSMMA_PDB", x_1, y_1)
    RFSMMA_PubChem_tpr = cross_validation(RFSMMA, "RFSMMA_PubChem", x_2, y_2)
    RFSMMA_RPocket_tpr = cross_validation(RFSMMA, "RFSMMA_RPocket", x_3, y_3)
    
    