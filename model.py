import collections
import itertools
import sys

from keras.layers import convolutional as cnn
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten
import librosa
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.cluster import KMeans


def audio_to_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=sr)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return duration, mfcc.reshape(tuple(reversed(mfcc.shape)))


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

def __to_time_sample(x):
    tmpl = "{:.0f}:{:.0f}-{:.0f}:{:.0f}"
    time_ = itertools.chain(divmod(x.sample[0], 60), divmod(x.sample[1], 60))
    return tmpl.format(*time_)

def make_consistent_samples(labels, track_duration, optimal_entropy):
    """Determine continuous consistent intervals of audio with the same tag.

    Could be needed for manual analysis."""
    sample = np.argmax(labels, 1)
    divide_idxs = [(0, len(sample)-1)]
    TimeFrame = collections.namedtuple("TimeFrame", ("tag", "sample"))
    consistent_list=[]
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
            frame_per_second = len(labels)//track_duration;
            time_frame=[idxs[0]//frame_per_second, idxs[1]//frame_per_second]
            item = TimeFrame(np.argmax(counts), time_frame)
            if item not in consistent_list and item.sample[1] - item.sample[0]:
                consistent_list.insert(0, item)

    consistent_list.sort()
    idx = 0
    while True:
        try:
            curr = consistent_list[idx]
            next_ = consistent_list[idx+1]
            if curr.tag == next_.tag and curr.sample[1] == next_.sample[0]:
                consistent_list[idx].sample[1] = consistent_list[idx+1].sample[1]
                del consistent_list[idx+1]
            else:
                idx += 1
        except IndexError:
            break

    consistent_list = sorted(consistent_list, key=lambda x: x.sample[0])
    return [TimeFrame(i.tag, __to_time_sample(i))  for i in consistent_list]


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


def main(train_track, prediction_track, instruments_num, test_track=None):
    """
    :param train_track: path to an audio file with known number of instruments in compositions
    :param train_track_annotaion: number of centroids
    :param test_track: path to an audio file with consistent instruments to test prediction.
    """
    _, train_mfccs = audio_to_mfcc(train_track)
    labels = tag_frames(train_mfccs, instruments_num)
    # model = build_model(64, train_mfccs.shape[1], labels.shape[1])
    model = build_cnn_model()#train_mfccs.shape[1])
    fit(model, train_mfccs, labels)
    if test_track:
        _,test_mfccs = audio_to_mfcc(test_track)
        print(model.evaluate(test_mfccs, labels, batch_size=32))
    duration, predict_mfccs = audio_to_mfcc(prediction_track)
    prediction = model.predict(predict_mfccs, batch_size=32)
    consistent_samples = make_consistent_samples(prediction, duration, 0.2)
    print(consistent_samples)

if __name__ == "__main__":
    # print(main("/Users/pgyschuk/Downloads/Within_Temptation_Ice_Queen.mp3", "/Users/pgyschuk/Downloads/Within_Temptation_Ice_Queen.mp3", 4)) #"/Users/pgyschuk/Downloads/Within_Temptation_Mother_Earth.mp3"
    train_track, prediction_track, instruments_num = sys.argv[1:]
    test_track = sys.argv[4] if len(sys.argv) == 5 else None
    main(train_track, prediction_track, int(instruments_num), test_track)
