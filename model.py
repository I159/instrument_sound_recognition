import collections
import itertools
import sys

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.cluster import KMeans


def build_model(output_dim, input_dim, tag_num):
    """Build keras model"""
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, init="normal", activation="relu"))
    model.add(Dense(output_dim, init="normal", activation="softmax"))
    model.add(Dense(tag_num, init="normal", activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def build_cnn_model(): #output_dim, input_dim, tag_num):
    model = Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        cnn.Convolution1D(nb_filter=64, filter_length=13, activation='relu', input_shape=(None, 13), batch_input_shape=(13,)),
        MaxPooling1D(),     # Downsample the output of convolution by 2X.
        cnn.Convolution1D(nb_filter=64, filter_length=13, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(4, activation='softmax'),     # For binary classification, change the activation to 'sigmoid'
    ))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # To perform (binary) classification instead:
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


def fit(model, mfccs, labels):
    model.fit(mfccs, labels, nb_epoch=5, batch_size=32)


def predict(model, mfccs):
    model.predict_classes(mfccs, batch_size=32)
