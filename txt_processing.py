import math
import os

import numpy as np
from pandas.io import parsers as ps


def parse_tags(tag_file_path):
    tags = ps.read_csv(tag_file_path)
    return tags["acoustic"]
    # rap = tag_list["acoustic"][tag_list["acoustic"] == "rap"]
    # rap.index[0]


def get_tagger(tags, train_map):
    def tagger(track_name):
        track_id = track_name.split(".")[0]
        track_tags = train_map.loc[train_map['id_song'] == int(track_id)]
        if track_tags.empty is False:
            tag_indexes = (tags[tags == i] for i in track_tags["tag"].values)
            tag_indexes = (i.index[0] for i in tag_indexes if not i.empty)
            hot_shot = np.zeros(len(tags))
            for i in tag_indexes:
                hot_shot[i] = 1
            return int(track_id), hot_shot
    return tagger


def get_mfcc_parser(dir_path):
    def mfcc_parser(track_id):
        file_path = os.path.join(dir_path, "{}.mp3.mfcc".format(track_id))
        return np.loadtxt(file_path, delimiter=",")
    return mfcc_parser


def get_standardizer(std_len):
    def reshape(mfcc_vector):
        pad_len = (std_len - mfcc_vector.shape[1]) / 2
        padded = np.lib.pad(
            mfcc_vector,
            ((0, 0), (int(pad_len), math.ceil(pad_len))),
            'constant',
            constant_values=(0, 0)
        )
        return padded.reshape(1, *padded.shape)
    return reshape


def get_mfcc_data_set(dir_path, tags, train_map):
    file_names = os.listdir(dir_path)
    tagger = get_tagger(tags, train_map)
    indexed_tags = [i for i in map(tagger, file_names) if i is not None]
    mfcc_parcer = get_mfcc_parser(dir_path)
    filtered_ids, tags = zip(*indexed_tags)

    mfccs = [mfcc_parcer(i) for i in filtered_ids]
    max_arr = math.ceil(max(mfccs, key=lambda x: x.shape[1]).shape[1]/1000)*1000
    standardizer = get_standardizer(max_arr)
    return np.array(tags), np.array([standardizer(i) for i in mfccs])
