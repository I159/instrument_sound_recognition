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


def build_cnn_model(): #output_dim, input_dim, tag_num):
    model = Sequential()

    # input image dimensions
    # rows, cols = 33, 70
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    model.add(Convolution1D(63, 3, border_mode='valid', input_shape=(None, 13)))
    model.add(Activation('sigmoid'))
    model.add(Convolution1D(63, 3))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.25))

    # model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(4)) # 4 classes
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model


def fit(model, mfccs, labels):
    model.fit(mfccs, labels, nb_epoch=5, batch_size=32)


def predict(model, mfccs):
    model.predict_classes(mfccs, batch_size=32)
