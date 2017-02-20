from collections import namedtuple
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")

import essentia
import essentia.standard as std
import essentia.streaming
from sklearn.cluster import KMeans
import numpy as np
from pylab import show, plot
from keras.models import Sequential
from keras.layers import Dense, Activation


def audio_to_mfcc(audio_path):
    loader = std.MonoLoader(filename=audio_path)
    audio = loader()

    w = std.Windowing(type='hann')
    spectrum = std.Spectrum()
    mfcc = std.MFCC()

    mfccs = []
    frameSize = 1024
    hopSize = 512

    for fstart in range(0, len(audio)-frameSize, hopSize):
        frame = audio[fstart:fstart+frameSize]
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)

    return mfccs


def tag_frames(mfccs):
    kmeans = KMeans(n_clusters=4, random_state=0).fit(mfccs)
    return kmeans.labels_


def make_consistent_samples(labels, track_duration):
    """Determine continuous consistent intervals of audio with the same tag.

    Could be needed for manual analysis."""
    track_duration = float(track_duration)
    frame_duration = track_duration / labels.shape[0]

    samples = []
    curr_sample = dict(label=None, start=None, end=None)
    for i, v in enumerate(labels):
        if curr_sample["label"] is None:
            curr_sample["label"] = v

        if v == curr_sample["label"]:
            if curr_sample["start"] is None:
                curr_sample["start"] = (i-1) * frame_duration if i-1 > 0 else 0
                curr_sample["end"] = i * frame_duration
            else:
                curr_sample["end"] = i * frame_duration
        else:
            samples.append((curr_sample["start"], curr_sample["end"]))
            curr_sample = dict(label=None, start=None, end=None)
    return samples


def build_model(output_dim=64, input_dim=128):
    """Build keras model"""
    model = Sequential()
    model.add(Dense(output_dim=output_dim, input_dim=input_dim))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def fit(model, mfccs, labels):
    model.fit(mfccs, labels, nb_epoch=5, batch_size=32)


def predict(model, mfccs):
    model.predict_classes(mfccs, batch_size=32)


def main(train_track, prediction_track, instruments_num, test_track=None):
    """
    :param train_track: path to an audio file with known number of instruments in compositions
    :param train_track_annotaion: number of centroids
    :param test_track: path to an audio file with consistent instruments to test prediction.
    """
    train_mfccs = audio_to_mfcc(train_track)
    labels = tag_frames(train_mfccs)
    model = build_model()
    fit(model, train_mfccs, labels)
    if test_track:
        print model.evaluate(test_track, labels, batch_size=32)
    return model.predict_classes(prediction_track, batch_size=32)

if __name__ == "__main__":
     print main(*sys.argv[1:])
