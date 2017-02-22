import pprint
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation
import librosa
import numpy as np
from sklearn.cluster import KMeans


def audio_to_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y, sr=sr)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return mfcc.reshape(tuple(reversed(mfcc.shape)))


def __one_hot_shot(num):
    def shot(idx):
        label = np.zeros(num)
        np.put(label, idx, 1)
        return label
    return shot


def tag_frames(mfccs, centroids_num):
    kmeans = KMeans(n_clusters=centroids_num, random_state=0).fit(mfccs)
    one_hot_shot = __one_hot_shot(centroids_num)
    return np.array([one_hot_shot(i) for i in kmeans.labels_])

def make_consistent_samples(labels, track_duration, optimal_entropy):
    """Determine continuous consistent intervals of audio with the same tag.

    Could be needed for manual analysis."""
    sample = np.argmax(labels, 1)
    divide_idxs = [(0, len(sample)-1)]
    consistent = [[], [], [], []]
    while divide_idxs:
        idxs = divide_idxs.pop()
        curr_sample = sample[idxs[0]: idxs[1]]
        counts = np.bincount(curr_sample)
        non_zero = counts[np.nonzero(counts)]
        prob = np.divide(non_zero, len(curr_sample))
        log2prod = np.vectorize(lambda x: x * np.log2(x))
        entropy = -sum(log2prod(prob))
        if entropy > optimal_entropy:
            median = ((idxs[1] - idxs[0]) // 2) + idxs[0]
            divide_idxs.append((idxs[0], median))
            divide_idxs.append((median, idxs[1]))
        else:
            consistent[np.argmax(counts)].append(idxs)
    return consistent


def build_model(output_dim, input_dim=None, input_shape=None):
    """Build keras model"""
    model = Sequential()
    model.add(Dense(13, input_dim=13, init="normal", activation="relu"))
    model.add(Dense(4, init="normal", activation="softmax"))
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
    labels = tag_frames(train_mfccs, instruments_num)
    model = build_model(output_dim=13)
    fit(model, train_mfccs, labels)
    if test_track:
        test_mfccs = audio_to_mfcc(test_track)
        print(model.evaluate(test_mfccs, labels, batch_size=32))
    predict_mfccs = audio_to_mfcc(prediction_track)
    prediction = model.predict(predict_mfccs, batch_size=32)
    consistent_samples = make_consistent_samples(prediction, 0, 0.4)
    pprint.pprint(consistent_samples)

if __name__ == "__main__":
    train_track, prediction_track, instruments_num = sys.argv[1:]
    test_track = sys.argv[4] if len(sys.argv) == 5 else None
    main(train_track, prediction_track, int(instruments_num), test_track)
