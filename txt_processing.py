import os
import re
import sys

import numpy as np
from pandas.io import parsers as ps


def parse_tags(tag_file_path):
    tags = ps.read_csv(tag_file_path)
    return tags["acoustic"]
    # rap = tag_list["acoustic"][tag_list["acoustic"] == "rap"]
    # rap.index[0]

def tag_tracks(tags, train_map):
    def tag_it(track_id):
        track_tags = train_map.loc[train_map['id_song'] == int(track_id)]
        if track_tags.empty is False:
            id_ = track_tags["id_song"].values[0]
            tag_indexes = [tags[tags == i].index[0] for i in track_tags["tag"].values]
            # TODO: one hot shot for tags
            return track_tags
    return tag_it


def parse_mfcc_txt(mfcc_file_path):
    return np.loadtxt(mfcc_file_path, delimiter=",")

def get_tagged_mfccs(dir_path, tags, train_map):
    file_paths = os.listdir(dir_path)
    ids = map(lambda x: x.split(".")[0], file_paths)
    paths = map(lambda x: os.path.join(dir_path, x), file_paths)
    tags = filter(lambda x: x is not None, map(tag_tracks(tags, train_map), ids))
    for i in tags:
        print(i)


def main():
    tag_path, mfcc_dir_path, train_map_path = sys.argv[1:]
    tags = parse_tags(tag_path)
    train_map = ps.read_csv(train_map_path, sep='\t')
    tagged_mfccs = get_tagged_mfccs(mfcc_dir_path, tags, train_map)
    # mfccs = parse_mfcc_txt(mfcc_path)
    # train_map = ps.read_csv(train_map_path, sep='\t')
    # track_id = re.findall(r"(\d+)\.mp3\.mfcc", mfcc_path)[0]
    # parse_track_tags(track_id, tags, train_map)


if __name__ == "__main__":
    main()
