from configparser import ConfigParser
import os
import re
import sys

import pandas as ps

from model import build_cnn_model, fit
from txt_processing import parse_tags, get_mfcc_data_set

CONFIG = ConfigParser()
CONFIG.readfp(open('paths.cfg'))


def main():
    tag_path, mfcc_dir_path, train_map_path = sys.argv[1:]
    tags = parse_tags(CONFIG.get("DATASETS", "tags_file"))
    train_map = ps.read_csv(CONFIG.get("DATASETS", "train_tags_file"), sep='\t')
    test_map = ps.read_csv(CONFIG.get("DATASETS", "test_tags_file"), sep='\t')

    train_tags, train_mfccs = get_mfcc_data_set(CONFIG.get("DATASETS", "mfcc_dir"), tags, train_map[:200])
    test_tags, test_mfccs = get_mfcc_data_set(CONFIG.get("DATASETS", "mfcc_dir"), tags, train_map[:100])

    model = build_cnn_model()
    fit(model, train_mfccs, train_tags, (test_mfccs, test_tags))

if __name__ == "__main__":
    main()
