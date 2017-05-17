from __future__ import print_function
import cv2
import numpy as np
import os
import time
import sys
from threading import Thread
from comm_publisher import Publisher
from comm_consumer import Consumer
from utils import *
import sparse_attention_model


class InitParams:
    def __init__(self):
        self.sequence = 'bigShipHighAlt_clip2'
        self.video_path = '/home/tiago/maritime_data_seq/' + self.sequence
        self.groundtruth_file = 'groundtruth_rect_detection.txt'

        self.command = 'python tracker_handler.py ' + self.video_path + ' '

        self.num_trackers = 0
        self.num_detections = 0
        self.tracker_counter = 0

        self.display = True


def main():
    params = InitParams()

    pub = Publisher('5551')
    sub1 = Consumer('5552', 'bb')
    sub2 = Consumer('5552', 'alive')

    # load ground truth file and images files
    groundtruth, img_seq = load_video_sequence(params.groundtruth_file, params.video_path)

    detections = []

    fp = open(params.sequence + '_output.txt', 'w')

    # iterate over every frame of the sequence selected
    for img_index in xrange(len(img_seq)):
        # time.sleep(0.1)

        img = cv2.imread(params.video_path + '/img/' + img_seq[img_index], 1)

        if img_index == 0:
            am = sparse_attention_model.AttentionModel(img)

        # full detection mode
        if params.num_trackers == 0:
            while groundtruth[params.num_detections][0].astype(np.int) == img_index:
                # print('img_index ({}) grd frame ({})'.format(img_index,
                #                                              groundtruth[params.num_detections]))

                if groundtruth[params.num_detections][6] > 0.95:
                    print("detection on frame", img_index)

                    # initialize tracker on previous detection
                    # TODO: what if there are multiple detections
                    bounding_box = ' '.join(map(str, groundtruth[params.num_detections][1:5].astype(np.int)))
                    detections = [0]
                    detections.extend(groundtruth[params.num_detections][1:5].astype(np.int))
                    detections = [detections]
                    # print(detections)
                    # print(params.command + bounding_box)

                    # TODO: Put in function
                    thread = Thread(target=threaded_function, args=((params.command + bounding_box + ' ' + str(params.tracker_counter)), ))
                    thread.start()

                    if sub2.recv_msg() == str(params.tracker_counter):
                        # print("sending index:", img_index)
                        params.num_detections += 1
                        params.num_trackers += 1
                        params.tracker_counter += 1
                        pub.send('next_img', str(img_index))

                params.num_detections += 1

        # using attention model
        else:
            roi = [0, 0, 0, 0]
            print("sending frame to trackers:", img_index)
            pub.send('next_img', str(img_index))

            try:
                # in this case the ground truth file contains a detection for the current frame
                # TODO: cannot be a while
                while groundtruth[params.num_detections][0].astype(np.int) == img_index:
                    # params.num_detections += 1

                    # perform detection on patch
                    roi = am.get_roi(detections, img)
                    # print(roi)
                    rst, id, bb = am.detect(roi, groundtruth[params.num_detections], detections)

                    if rst == 'UPDATE':
                        print("UPDATE")
                        # print("tracker counter:", params.tracker_counter)
                        pub.send('update', id)
                        pub.send('next_img', '0')
                        sub2.recv_msg()
                        # time.sleep(0.2)
                        thread = Thread(target=threaded_function, args=((params.command + bb + ' ' + str(params.tracker_counter)),))
                        thread.start()

                        if sub2.recv_msg() == str(params.tracker_counter):
                            # print("sending index:", img_index)
                            params.num_detections += 1
                            params.tracker_counter += 1
                            pub.send('next_img', str(img_index))

                    elif rst == 'INSERT':
                        print("INSERT")
                        thread = Thread(target=threaded_function, args=((params.command + bb + ' ' + str(params.tracker_counter)),))
                        thread.start()

                        if sub2.recv_msg() == str(params.tracker_counter):
                            # print("sending index:", img_index)
                            params.num_detections += 1
                            params.tracker_counter += 1
                            pub.send('next_img', str(img_index))
                    elif rst == 'DELETE':
                        pass
                    else:
                        pass

                    params.num_detections += 1

                # no detection for the whole frame
                # else:
                #     print("No detection for frame:", img_index)
            except:
                break

        detections = []
        for i in xrange(params.num_trackers):
            aux = sub1.recv_msg()
            print("recv det", aux)
            aux = aux.split(' (', 1)
            entry = [int(aux[0])]
            aux = aux[1][: -1]
            entry.extend([int(s) for s in aux.split(', ')])
            detections.append(entry)
            # print("received detection:", detections)

        if params.display:
            # display image
            for i in xrange(params.num_trackers):
                cv2.rectangle(img, (detections[i][1], detections[i][2]),
                              (detections[i][1] + detections[i][3], detections[i][2] + detections[i][4]), (0, 255, 0), 2)
            try:
                top_left = (roi[0], roi[1])
                bottom_right = (roi[0] + roi[2], roi[1] + roi[3])
                # print("top bottom:", top_left, bottom_right)
                cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
            except:
                pass
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", int(img.shape[1]/2), int(img.shape[0]/2))
            cv2.imshow("Image", img)
            cv2.waitKey(1)

        # in case a detection goes outside image
        # TODO: make it work for multiple detections
        if len(detections) > 0:
            if detections[0][1] + detections[0][3]/2 < 0 or detections[0][2] + detections[0][4]/2 < 0 or detections[0][5] < 7:
                print("LOST")
                # print(detections[0][0])
                pub.send('next_img', '0')
                pub.send('kill', str(detections[0][0]))
                params.num_trackers -= 1
                # params.tracker_counter += 1
                sub2.recv_msg()
                print("received confirmation")


            fp.write(str(img_index) + ' ' + str(detections[0][1:-1]) + '\n')

    pub.send('kill', 'all')
    fp.close()
    print("bye, i'm out!")


if __name__ == "__main__":
    main()
