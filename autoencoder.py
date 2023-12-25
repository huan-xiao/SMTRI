# to train an autoencoder to generate latent features for RNA motifs
# latent space: 512

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# set network seed to make it reproducible
keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()


def load_data():
    dict = {'A': [1,0,0,0,0], 'C': [0,1,0,0,0], 'G': [0,0,1,0,0], 'U': [0,0,0,1,0]}
    
    # read unique RNA motif sequences
    df = pd.read_csv('./data/RNA_motif_sequences.csv')
    
    motifs = []
    for line in df['motif']:
        motif=[]
        for letter in line:
            motif.append(dict[letter])
        motifs.append(motif)

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(motifs, maxlen=16,padding="post",value=[0,0,0,0,1])

    X_train, X_test = train_test_split(padded_inputs, test_size=0.1, random_state=0)
    
    return X_train, X_test



def print_train_loss(history, dimension):
    plt.clf()
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(0,dimension)
    plt.plot(epochs, loss_train[0:], 'g', label='Training loss')
    plt.plot(epochs, loss_val[0:], 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./data/AE training and validation loss.png")
    
    

def print_train_accuracy(history, dimension):
    plt.clf()
    loss_train = history.history['categorical_accuracy']
    loss_val = history.history['val_categorical_accuracy']
    epochs = range(0,dimension)
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("./data/AE training and validation accuracy.png") 
    
    
    
def save_model():
    saved_model = tf.keras.models.load_model('./model/best_AE_model.h5')
    # change model to encoder output
    model_encoder = Model(inputs=saved_model.inputs, outputs=saved_model.layers[2].output)
    model_encoder.compile(optimizer='Adamax', loss=keras.losses.CategoricalCrossentropy()) # if don't compile, error when loaded
    print(model_encoder.summary())
    model_encoder.save('./model/autoencoder.h5')
    
    

if __name__ == "__main__":
    
    # define LSTM autoencoder: parameters
    timesteps = 16
    input_dim = 5

    # define LSTM autoencoder: models
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(128, return_sequences=True)(inputs)
    encoded = LSTM(512, return_sequences=False)(encoded)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(512, return_sequences=True)(decoded)
    decoded = LSTM(128, return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(5, activation = 'softmax', use_bias=False))(decoded)
    
    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    # define loss function to be CategoricalCrossentropy
    cce = keras.losses.CategoricalCrossentropy()
    sequence_autoencoder.compile(optimizer='Adamax', loss=cce, metrics=[keras.metrics.CategoricalAccuracy()])
    mc = ModelCheckpoint('./model/best_AE_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
 
    print(sequence_autoencoder.summary())

    X_train, X_test = load_data()
    # fit model
    dimension = 80 # epochs
    history = sequence_autoencoder.fit(X_train, X_train, batch_size=32, epochs=dimension, verbose=1,validation_data=(X_test, X_test),callbacks=[mc])

    print_train_loss(history, dimension)
    print_train_accuracy(history, dimension)
    save_model()