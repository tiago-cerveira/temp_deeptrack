import numpy as np
import os
import sys


def load_video_sequence(groundtruth_file=None, video_path=None):
    if groundtruth_file is not None:
        groundtruth = np.loadtxt(video_path + '/' + groundtruth_file, delimiter=" ")
        img_seq = sorted([f for f in os.listdir(video_path + '/img') if os.path.isfile(os.path.join(video_path + '/img', f))])
        return groundtruth, img_seq
    else:
        img_seq = sorted([f for f in os.listdir(video_path + '/img') if os.path.isfile(os.path.join(video_path + '/img', f))])
        return img_seq


def threaded_function(arg):
    # print arg
    os.system(arg)


def get_video_path():
    return sys.argv[1]


def get_bounding_box():
    return sys.argv[2:6]