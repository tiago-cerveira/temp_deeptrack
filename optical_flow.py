import numpy as np
import cv2
from utils import *
import time

video_path = '/home/tiago/maritime_data_seq/lanchaArgos_clip3'

img_seq = load_video_sequence(video_path=video_path)

# cap = cv2.VideoCapture('slow.flv')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

old_frame = cv2.imread(video_path + '/img/' + img_seq[0], 1)
j = 0
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
# ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(j < 500):
    print j
    j += 1
    frame = cv2.imread(video_path + '/img/' + img_seq[j+80], 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    max = np.max(flow)
    cv2.imshow('1', flow[:, :, 0]/max)
    cv2.waitKey(-1)
    # Select good points
    try:
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        cv2.waitKey(100)
        print a-c, b-d
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    except:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # time.sleep(2)
cv2.destroyAllWindows()
