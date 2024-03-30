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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import cohen_kappa_score

keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
np.random.seed(0)


def print_auc(model_name, y_test, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, thresholds = roc_curve(y_test, abs(y_pred))# use results instead of predictions
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.rcParams.update({'font.size': 14})
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc, alpha=0.9)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.7)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./result/'+model_name+'_ROC_AUC.png')
    plt.close()
    
    
def print_confusion_matrix(model_name, y_test, predictions):
    class_names = [0,1]
    cm = confusion_matrix(y_test, predictions, labels=class_names)
    plt.figure()
    plt.rcParams.update({'font.size': 14})
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N','P'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('./result/'+model_name+'_Confusion_Matrix.png')
    plt.close()
    
    
def print_metrics(model_name, y_test, y_pred):
    
    print_auc(model_name, y_test, y_pred)

    predictions = np.round(y_pred)
    print_confusion_matrix(model_name, y_test, predictions)
    
    print("################# "+model_name+" #################")
    print(classification_report(y_test, predictions,digits=3))
    print('Accuracy:', end=' ')
    print(round(accuracy_score(y_test, predictions),3))
    print('MCC:', end=' ')
    print(round(matthews_corrcoef(y_test, predictions),3))
    print('Kappa:', end=' ')
    print(round(cohen_kappa_score(y_test, predictions),3))

        

    
if __name__ == "__main__":
    
    # load data
    training_set = pd.read_csv('./data/Training_set.csv')
    x_train = np.array(training_set.iloc[:,:-1])
    y_train = np.array(training_set.iloc[:,-1:]).reshape(-1)
    x_train, y_train = shuffle(x_train, y_train, random_state=1)
    
    
    # three testing datasets
    testing_set_1 = pd.read_csv('./data/PDB_testing_set.csv')
    x_test_1 = np.array(testing_set_1.iloc[:,:-1])
    y_test_1 = np.array(testing_set_1.iloc[:,-1:]).reshape(-1)
    x_test_1, y_test_1 = shuffle(x_test_1, y_test_1, random_state=1)
    
    testing_set_2 = pd.read_csv('./data/PubChem_testing_set.csv')
    x_test_2 = np.array(testing_set_2.iloc[:,:-1])
    y_test_2 = np.array(testing_set_2.iloc[:,-1:]).reshape(-1)
    x_test_2, y_test_2 = shuffle(x_test_2, y_test_2, random_state=1)
    
    testing_set_3 = pd.read_csv('./data/RPocket_testing_set.csv')
    x_test_3 = np.array(testing_set_3.iloc[:,:-1])
    y_test_3 = np.array(testing_set_3.iloc[:,-1:]).reshape(-1)
    x_test_3, y_test_3 = shuffle(x_test_3, y_test_3, random_state=1)
    
    
    
    # SMTRI
    SMTRI = tf.keras.models.load_model('./model/best_DNN_model.h5')
    y_pred_1 = SMTRI.predict(x_test_1, verbose=0)
    print_metrics('SMTRI_PDB', y_test_1, y_pred_1)
    y_pred_2 = SMTRI.predict(x_test_2, verbose=0)
    print_metrics('SMTRI_PubChem', y_test_2, y_pred_2)
    y_pred_3 = SMTRI.predict(x_test_3, verbose=0)
    print_metrics('SMTRI_RPocket', y_test_3, y_pred_3)
    
    
    # XGBoost
    XGB = XGBClassifier(objective= 'binary:logistic', n_estimators= 200, min_child_weight= 20, max_depth= 20, learning_rate= 0.01, gamma= 2, colsample_bytree= 0.8, booster= 'gbtree').fit(x_train, y_train)
    
    y_pred_1 = XGB.predict_proba(x_test_1)[:,1]
    print_metrics('XGB_PDB', y_test_1, y_pred_1)
    y_pred_2 = XGB.predict_proba(x_test_2)[:,1]
    print_metrics('XGB_PubChem', y_test_2, y_pred_2)
    y_pred_3 = XGB.predict_proba(x_test_3)[:,1]
    print_metrics('XGB_RPocket', y_test_3, y_pred_3)
    
    
    # NB
    NB = GaussianNB(var_smoothing=1e-8).fit(x_train, y_train)

    y_pred_1 = NB.predict_proba(x_test_1)[:,1]
    print_metrics('NB_PDB', y_test_1, y_pred_1)
    y_pred_2 = NB.predict_proba(x_test_2)[:,1]
    print_metrics('NB_PubChem', y_test_2, y_pred_2)
    y_pred_3 = NB.predict_proba(x_test_3)[:,1]
    print_metrics('NB_RPocket', y_test_3, y_pred_3)
    
    
    # RFSMMA
    RFSMMA = RandomForestClassifier(n_estimators= 25, min_samples_split= 4, max_features= 'log2', max_depth= 10, criterion= 'log_loss').fit(x_train, y_train)
    
    y_pred_1 = RFSMMA.predict_proba(x_test_1)[:,1]
    print_metrics('RFSMMA_PDB', y_test_1, y_pred_1)
    y_pred_2 = RFSMMA.predict_proba(x_test_2)[:,1]
    print_metrics('RFSMMA_PubChem', y_test_2, y_pred_2)
    y_pred_3 = RFSMMA.predict_proba(x_test_3)[:,1]
    print_metrics('RFSMMA_RPocket', y_test_3, y_pred_3)
    
    
    
    