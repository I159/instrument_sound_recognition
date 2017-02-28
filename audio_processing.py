from keras.layers import convolutional as cnn
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten
import librosa
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.cluster import KMeans


def audio_to_mfcc(audio_path):
    y, sr = librosa.load(audio_path, mono=False)
    duration = librosa.get_duration(y=y, sr=sr)
    S1 = librosa.feature.melspectrogram(y[0], sr=sr)
    S2 = librosa.feature.melspectrogram(y[1], sr=sr)
    log_S1 = librosa.logamplitude(S1, ref_power=np.max)
    log_S2 = librosa.logamplitude(S2, ref_power=np.max)
    mfcc1 = librosa.feature.mfcc(S=log_S1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(S=log_S2, n_mfcc=13)

    mfcc = np.dstack((mfcc1, mfcc2))
    return duration, mfcc.reshape((mfcc.shape[1], mfcc.shape[2], mfcc.shape[0]))


def __one_hot_shot(num):
    def shot(idx):
        label = np.zeros(num)
        np.put(label, idx, 1)
        return label
    return shot


def tag_frames(mfccs, centroids_num):
    clustering = KMeans(n_clusters=centroids_num, random_state=0)
    kmeans1 = clustering.fit(mfccs[0:,0])
    kmeans2 = clustering.fit(mfccs[0:,1])
    one_hot_shot = __one_hot_shot(centroids_num)
    hot_labels1 = np.array([one_hot_shot(i) for i in kmeans1.labels_])
    hot_labels2 = np.array([one_hot_shot(i) for i in kmeans2.labels_])
    labels = np.dstack((hot_labels1, hot_labels2))
    return labels.reshape((labels.shape[0], labels.shape[2], labels.shape[1]))


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
