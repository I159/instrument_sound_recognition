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

def __to_mfcc_coeff(frame):
    window = std.Windowing(type='hann')
    spectrum = std.Spectrum()
    mfcc = std.MFCC()

    return mfcc(spectrum(window(frame)))[1]


def audio_to_mfcc(audio_path):
    loader = std.MonoLoader(filename=audio_path)
    audio = loader()

    frame_gen = std.FrameGenerator(audio, frameSize=2048, hopSize=512)
    mfcc_array = essentia.array(map(__to_mfcc_coeff, frame_gen)).T
    return mfcc_array.reshape(tuple(reversed(mfcc_array.shape)))


def tag_frames(mfccs, centroids_num):
    kmeans = KMeans(n_clusters=centroids_num, random_state=0).fit(mfccs)
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


def build_model(output_dim, input_dim=None, input_shape=None):
    """Build keras model"""
    model = Sequential()
    model.add(Dense(output_dim=13, input_dim=13))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=13))
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
    train_mfccs = np.zeros((256, 13))#audio_to_mfcc(train_track)
    labels = np.zeros((256, 13))#tag_frames(train_mfccs, instruments_num)
    model = build_model(output_dim=13)
    fit(model, train_mfccs, labels)
    if test_track:
        print model.evaluate(test_track, labels, batch_size=32)
    return model.predict_classes(prediction_track, batch_size=32)

if __name__ == "__main__":
     train_track, prediction_track, instruments_num = sys.argv[1:]
     test_track = sys.argv[4] if len(sys.argv) == 5 else None
     print main(train_track, prediction_track, int(instruments_num), test_track)
