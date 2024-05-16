# the core model for SMTRI

import numpy as np
from numpy import loadtxt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from scipy import interp

# set network seed to make it reproducible
keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()


def print_train_loss(history, dimension):
    plt.clf()
    lower_bound = 0
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(lower_bound,dimension)
    plt.plot(epochs, loss_train[lower_bound:], 'g', label='Training loss')
    plt.plot(epochs, loss_val[lower_bound:], 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./result/CNN training and validation loss.png")
    
    
def print_train_accuracy(history, dimension):
    plt.clf()
    lower_bound = 0
    loss_train = history.history['binary_accuracy']
    loss_val = history.history['val_binary_accuracy']
    epochs = range(lower_bound,dimension)
    plt.plot(epochs, loss_train[lower_bound:], 'g', label='Training accuracy')
    plt.plot(epochs, loss_val[lower_bound:], 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("./result/CNN training and validation accuracy.png") 

    
def print_auc(model_name, y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, abs(y_pred))# use results instead of predictions
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc, alpha=0.9)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.7)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./result/'+model_name+'_ROC_AUC.png')
    
     
def print_confusion_matrix(model_name, y_test, predictions):
    
    class_names = [0,1]

    cm = confusion_matrix(y_test, predictions, labels=class_names)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N','P'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('./result/'+model_name+'SMTRI_Confusion_Matrix.png')
    
    
def print_metrics(model_name, y_test, y_pred):
    
    predictions = np.round(y_pred)
    print_confusion_matrix(model_name, y_test, predictions)
    print(classification_report(y_test, predictions,digits=3))
    print('AUC:', end=' ')
    print(round(roc_auc_score(y_test, y_pred),3))
    print('Accuracy:', end=' ')
    print(round(accuracy_score(y_test, predictions),3))
    print('Precision:', end=' ')
    print(round(precision_score(y_test, predictions),3))
    print('Recall:', end=' ')
    print(round(recall_score(y_test, predictions),3))
    print('F1-Score:', end=' ')
    print(round(f1_score(y_test, predictions),3))
    print('MCC:', end=' ')
    print(round(matthews_corrcoef(y_test, predictions),3))
    print('Kappa:', end=' ')
    print(round(cohen_kappa_score(y_test, predictions),3))
    
    

def CNN_model():
    # model layers
    inputs = Input(shape=(1620,1))
    cov1 = Conv1D(3, 3, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    bn_max1 = BatchNormalization()(cov1)
    maxpool1 = MaxPool1D(data_format='channels_first')(bn_max1)
    cov2 = Conv1D(2, 2, activation='relu', kernel_regularizer=l2(0.01))(maxpool1)
    bn_max2 = BatchNormalization()(cov2)
    maxpool2 = MaxPool1D(data_format='channels_first')(bn_max2)
    cov3 = Conv1D(2, 2, activation='relu', kernel_regularizer=l2(0.01))(maxpool2)
    bn_max3 = BatchNormalization()(cov3)
    maxpool3 = GlobalMaxPooling1D(data_format='channels_first')(bn_max3)

    dnn1 = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(maxpool3)
    bn1 = BatchNormalization()(dnn1)
    dnn2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(bn1)
    bn2 = BatchNormalization()(dnn2)
    dnn3 = Dense(8, activation='relu', kernel_regularizer=l2(0.01))(bn2)
    bn3 = BatchNormalization()(dnn3)
    outputs = Dense(1, activation='sigmoid')(bn3)
    model = Model(inputs=inputs, outputs=outputs)

    # compile the keras model binary_crossentropy,BinaryCrossentropy()
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['binary_accuracy'])
    
    return model
    
    
def training(model, x_train, y_train):
    dimension = 78
    history = model.fit(x_train, y_train, batch_size=32, epochs=dimension, verbose=1)
    

def training_with_CV(model, x_train, y_train):
    n_splits=5
    cv = StratifiedKFold(n_splits, shuffle=True)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['red','orange','yellow','green','blue']
    
    for fold, (train, test) in enumerate(cv.split(x_train, y_train)):
        model = CNN_model()
        model.fit(x_train[train], y_train[train], epochs=78, batch_size=32, verbose=1)
        y_pred = model.predict(x_train[test], verbose=0)

        viz = RocCurveDisplay.from_predictions(
            y_train[test],
            y_pred,
            name=f"ROC fold {fold}",
            alpha=0.3,
            color = colors[fold],
            ax=ax
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.7)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Receiver Operating Characteristic",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    
    plt.savefig('./result/SMTRI_CV_ROC.png')
    
    
if __name__ == "__main__":
    
    # load train and test feature data
    x_train = np.load('./data/x_train.npy')
    x_test = np.load('./data/x_test.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')
    
    # define model
    model = CNN_model()
    print(model.summary())
    
    # train with CV=5, to tune the parameters
    #training_with_CV(model, x_train, y_train)
    
    # train CNN model, with best parameters
    training(model, x_train, y_train)
    model.save('./model/best_DNN_model.h5')
    
    # make prediction
    saved_model = tf.keras.models.load_model('./model/best_DNN_model.h5')
    y_pred = saved_model.predict(x_test, verbose=0)
    
    # evaluate the model
    print_metrics("SMTRI", y_test, y_pred)
    print_auc("SMTRI", y_test, y_pred)
    
    