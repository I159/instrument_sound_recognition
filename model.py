import collections
import itertools
import sys

import numpy as np
from keras.layers import convolutional as cnn
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Convolution1D, MaxPooling1D

from sklearn.cluster import KMeans


def build_model(output_dim, input_dim, tag_num):
    """Build keras model"""
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, init="normal", activation="relu"))
    model.add(Dense(output_dim, init="normal", activation="softmax"))
    model.add(Dense(tag_num, init="normal", activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def build_cnn_model():
    model = Sequential()

    model.add(Convolution2D(3, 1, 20, border_mode='valid', input_shape=(1, 20, 33000)))
    model.add(Activation('sigmoid'))
    model.add(Convolution2D(3, 1, 1))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling2D(pool_size=(1, 1), dim_ordering="tf"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(93))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model


def fit(model, mfccs, labels, validation_data):
    model.fit(mfccs, labels, nb_epoch=25, batch_size=4, validation_split=0.33, validation_data=validation_data)
    metrics = model.evaluate(*validation_data, verbose=0)


def predict(model, mfccs):
    return model.predict(mfccs, batch_size=4)
