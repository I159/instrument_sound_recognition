import librosa
import sys
from sklearn.cluster import KMeans
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

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
    return np.array(map(__one_hot_shot(centroids_num), kmeans.labels_))


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
    model.add(Dense(13, input_dim=13, init="normal", activation="relu"))
    model.add(Dense(13, input_dim=13, init="normal", activation="relu"))
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
    return prediction

if __name__ == "__main__":
    train_track, prediction_track, instruments_num = sys.argv[1:]
    test_track = sys.argv[4] if len(sys.argv) == 5 else None
    print(main(train_track, prediction_track, int(instruments_num), test_track))
