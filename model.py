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
    # model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
    # model.add(Activation('sigmoid'))
    # model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    # model.add(Activation('sigmoid'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())

    # model.add(Dense(128))
    # model.add(Activation('sigmoid'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4)) # 4 classes
    # model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='adadelta')
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
    model.add(Dense(93)) # 4 classes
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model


def fit(model, mfccs, labels):
    model.fit(mfccs, labels, nb_epoch=25, batch_size=4)


def predict(model, mfccs):
    model.predict_classes(mfccs, batch_size=32)
