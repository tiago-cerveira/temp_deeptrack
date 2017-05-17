from __future__ import print_function
import cv2
import sys
import pika
from time import sleep
import numpy as np
from comm_consumer import Consumer
from comm_publisher import Publisher
import zmq
from utils import *
from tracker import Tracker
import time


class InitParams:
    def __init__(self):
        self.padding = 1.5                              # extra area surrounding the target
        self.output_sigma_factor = 0.1                  # standard deviation for the desired translation filter output
        self.lmbda = 1e-4                               # regularization weight
        self.learning_rate = 0.02

        self.scale_sigma_factor = 1.0/4                 # standard deviation for the desired scale filter output
        self.number_of_scales = 1                       # number of scale levels
        self.scale_step = 1.02                          # Scale increment factor
        self.scale_model_max_area = 512                 # maximum scale

        self.features = "CNN_TF"
        self.cell_size = 4.0
        self.high_freq_threshold = 2 * 10 ** 66
        self.peak_to_sidelobe_ratio_threshold = 6       # Set to 0 to disable (Detect if the target is lost)
        self.rigid_transformation_estimation = False    # Try to detect camera rotation

        self.visualization = False
        self.debug = False

        self.init_pos = np.array((0, 0))
        self.pos = np.array((0, 0))
        self.target_size = np.array((0, 0))
        self.img_files = None
        self.video_path = None

        self.kernel = Kernel()

        self.frame = 0

        self.id = ''


# Structure with kernel parameters
class Kernel:
    def __init__(self):
        self.kernel_type = "Linear"  # Or Gaussian
        self.kernel_sigma = 0.5


def main():

    params = InitParams()

    video_path = get_video_path()
    img_seq = load_video_sequence(groundtruth_file=None, video_path=video_path)
    bounding_box = get_bounding_box()
    # print("i'm the tracker and received:", bounding_box)

    params.id = sys.argv[6]

    # Initial position
    pos = np.array([int(bounding_box[1]), int(bounding_box[0])])
    target_sz = np.array([int(bounding_box[3]), int(bounding_box[2])])
    params.init_pos = np.floor(pos + np.floor(target_sz / 2))

    # Current position
    params.pos = params.init_pos

    # Size of target
    params.target_size = np.floor(target_sz)
    params.img_files = img_seq

    # List of image files
    params.video_path = video_path

    # results = np.zeros((len(img_seq), 4))
    # time.sleep(0.5)
    pub = Publisher('5552')
    sub1 = Consumer('5551', 'next_img')
    sub2 = Consumer('5551', 'kill')
    sub3 = Consumer('5551', 'update')

    # this need to go!
    time.sleep(0.4)
    pub.send('alive', params.id)

    starting = True

    print("new tracker is ready to consume messages")
    while True:

        img_index = int(sub1.recv_msg())
        # print(img_index)
        # print(video_path + '/img/' + img_seq[img_index])
        img = cv2.imread(video_path + '/img/' + img_seq[img_index], 1)
        msg = sub2.recv_msg_no_block()
        if msg is not False:
            split_msg = msg.split()
            if split_msg[0] == params.id or split_msg[0] == 'all':
                print("tracker " + params.id + " received kill command, now exiting")
                break

        msg = sub3.recv_msg_no_block()
        if msg is not False:
            split_msg = msg.split()
            if split_msg[0] == params.id:
                print("tracker received update command, now exiting")
                break
                # print("SHOULD BE UPDATING")
                # params.frame = 0
                #
                # # Initial position
                # pos = np.array([int(split_msg[2]), int(split_msg[1])])
                # target_sz = np.array([int(split_msg[4]), int(split_msg[3])])
                # # params.init_pos = np.floor(pos + np.floor(target_sz / 2))
                #
                # # Current position
                # params.pos = params.init_pos
                #
                # # Size of target
                # # params.target_size = np.floor(target_sz)
                #
                # tracker1.update_tracker((int(split_msg[1]), int(split_msg[2])), (int(split_msg[3]), int(split_msg[4])))

        # Initialize the tracker using the first frame
        if params.frame == 0:
            if starting:
                tracker1 = Tracker(img, params)
                starting = False
            tracker1.train(img, True)
            results = np.array(
                (pos[0] + np.floor(target_sz[0] / 2), pos[1] + np.floor(target_sz[1] / 2), target_sz[0], target_sz[1], 10))
        else:
            results, lost, xtf = tracker1.detect(img)  # Detect the target in the next frame
            if not lost:
                tracker1.train(img, False, xtf)  # Update the model with the new information
        # print(results)
        cvrect = np.array((results[1] - results[3] / 2,
                           results[0] - results[2] / 2,
                           results[1] + results[3] / 2,
                           results[0] + results[2] / 2))

        if params.visualization:
            # Draw a rectangle in the estimated location and show the result

            cv2.rectangle(img, (cvrect[0].astype(int), cvrect[1].astype(int)),
                          (cvrect[2].astype(int), cvrect[3].astype(int)), (0, 255, 0), 2)
            cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Window", int(img.shape[1] / 2), int(img.shape[0] / 2))
            cv2.imshow('Window', img)
            cv2.waitKey(1)

        params.frame += 1
        # print("tracker side:", str((cvrect).astype(np.int)))
        if params.frame == 1 or params.frame == 2:
            pub.send('bb', params.id + ' ' + str(
                (cvrect[0].astype(int), cvrect[1].astype(int), target_sz[1], target_sz[0], 10)))
        else:
            pub.send('bb', params.id + ' ' + str((cvrect[0].astype(int), cvrect[1].astype(int), target_sz[1], target_sz[0], int(results[4]))))

    # np.savetxt('results.txt', results, delimiter=',', fmt='%d')
    pub.send('alive', 'False')
    pub.close()
    sub1.close()
    sub2.close()
    sub3.close()


if __name__ == "__main__":
    main()