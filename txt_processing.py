from pandas.io import import parsers as ps


def parse_tags(tag_file_path):
    tags = ps.read_csv(tag_file_path)
    return tags["acoustic"]
    # rap = tag_list["acoustic"][tag_list["acoustic"] == "rap"]
    # rap.index[0]

def parse_mfcc_txt(mfcc_file_path):
    return np.loadtxt(mfccs_file, delimiter=",")
