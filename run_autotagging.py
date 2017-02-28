import sys

from audio_processing import audio_to_mfcc, tag_frames, make_consistent_samples
from model import build_cnn_model, fit


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
