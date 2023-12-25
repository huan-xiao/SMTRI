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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
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
    plt.savefig("./data/CNN training and validation loss.png")
    
    
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
    plt.savefig("./data/CNN training and validation accuracy.png") 

    
def print_auc(y_test, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

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
    plt.savefig('./data/ROC_AUC.png')
    
     
def print_confusion_matrix(y_test, predictions):
    
    class_names = [0,1]

    cm = confusion_matrix(y_test, predictions, labels=class_names)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N','P'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('./data/Confusion_Matrix.png')
    
    
def print_metrics(x_test, y_test):
    saved_model = tf.keras.models.load_model('./model/best_DNN_model.h5')
    
    y_pred = saved_model.predict(x_test, verbose=0)
    print_auc(y_test, y_pred)
    
    predictions = np.round(y_pred)
    print_confusion_matrix(y_test, predictions)
    print(classification_report(y_test, predictions,digits=3))
    print('Accuracy:', end=' ')
    print(round(accuracy_score(y_test, predictions),3))
    print('MCC:', end=' ')
    print(round(matthews_corrcoef(y_test, predictions),3))
    
    
def CNN_based_model(x_train, x_test, y_train, y_test):
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
    
    dimension = 78
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=dimension, verbose=1)
    
    # print train loss and accuracy
    #print_train_loss(history, dimension)
    #print_train_accuracy(history, dimension)
    
    model.save('./model/best_DNN_model.h5')
    
    
if __name__ == "__main__":
    # read feature data, split to train and test sets
    '''
    training_set = pd.read_csv('./data/training_set.csv')
    
    X=training_set.iloc[:, 0:-1]
    y=training_set['target']
    
    X = X.to_numpy()
    y = y.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    np.save('./data/x_train.npy', x_train)
    np.save('./data/x_test.npy', x_test)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/y_test.npy', y_test)
    '''
    
    # load train and test feature data
    x_train = np.load('./data/x_train.npy')
    x_test = np.load('./data/x_test.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')
    
    # train CNN model
    CNN_based_model(x_train, x_test, y_train, y_test)
    
    # evaluate the model
    print_metrics(x_test, y_test)