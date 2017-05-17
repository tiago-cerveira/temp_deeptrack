from __future__ import print_function
import time
import numpy as np


class ROI:
    # max uncertainty (1 - boat, 0 - no boat)
    uncertainty = 0.5
    slope = 0.005
    detected, undetected = False, False

    def __init__(self, center, window_sz, top=False, bottom=False, left=False, right=False):
        self.center = center
        self.window_sz = window_sz
        # print(self.center)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def update_uncertainty(self, new_uncertainty):
        if self.detected:
            self.uncertainty = 1
            self.detected = False
        elif self.undetected:
            self.uncertainty = 0
            self.undetected = False
        else:
            self.uncertainty = new_uncertainty
            self.uncertainty -= self.func(self.slope)

    def func(self, slope):
        return 2 * slope * (self.uncertainty - 0.5)

    def arg(self):
        return -4*(self.uncertainty-0.5)**2 + 1


def create_rois(ncols, nrows, window_sz):
    rois = []
    for i in xrange(ncols):
        for j in xrange(nrows):
            # print(i, j)
            x = int(i*window_sz[0] + window_sz[0]/2)
            y = int(j*window_sz[1] + window_sz[1]/2)
            if i == 0 and j == 0:
                # print('top left', end=' ')
                rois.append(ROI((x, y), window_sz, top=True, left=True))
            elif i == ncols-1 and j == nrows-1:
                # print('bottom right', end=' ')
                rois.append(ROI((x, y), window_sz, bottom=True, right=True))
            elif i == 0 and j == nrows-1:
                # print('bottom left', end=' ')
                rois.append(ROI((x, y), window_sz, bottom=True, left=True))
            elif i == ncols-1 and j == 0:
                # print('top right', end=' ')
                rois.append(ROI((x, y), window_sz, top=True, right=True))
            elif i == 0 and j != 0:
                # print('left', end=' ')
                rois.append(ROI((x, y), window_sz, left=True))
            elif i != 0 and j == 0:
                # print('top', end=' ')
                rois.append(ROI((x, y), window_sz, top=True))
            elif i == ncols-1 and j != nrows-1:
                # print('right', end=' ')
                rois.append(ROI((x, y), window_sz, right=True))
            elif i != ncols-1 and j == nrows-1:
                # print('bottom', end=' ')
                rois.append(ROI((x, y), window_sz, bottom=True))
            else:
                rois.append(ROI((x, y), window_sz))
            # time.sleep(2)

    return rois


def area(rect_a, rect_b):
    dx = min(rect_a.center[0]+rect_a.window_sz[0]/2, rect_b.center[0]+rect_b.window_sz[0]/2) - max(rect_a.center[0]-rect_a.window_sz[0]/2, rect_b.center[0]-rect_b.window_sz[0]/2)
    dy = min(rect_a.center[1]+rect_a.window_sz[1]/2, rect_b.center[1]+rect_b.window_sz[1]/2) - max(rect_a.center[1]-rect_a.window_sz[1]/2, rect_b.center[1]-rect_b.window_sz[1]/2)
    if dx > 0 and dy > 0:
        return dx*dy


def normalize_prop(uncert_prop):
    for pos in uncert_prop:
        # too much info, need to normalize
        if pos[1] > 1:
            pos[0] /= pos[1]
        # too litle propagation, uncertainty arises
        elif pos[1] < 1:
            pos[0] = pos[0] * pos[1] + 0.5 * (1 - pos[1])
    return uncert_prop


def prop_uncertainty(rois, flow, window_sz, uncert_prop):
    for roi in rois:
        # mov_center = flow[roi.center[1], roi.center[0], :] # just for the centers
        mov_center = flow[roi.center[1]-window_sz[1]/2:roi.center[1]+window_sz[1]/2, roi.center[0]-window_sz[0]/2:roi.center[0]+window_sz[0]/2, :].mean(axis=(0,1), dtype=np.float64)
        mov_center = np.flip(mov_center, 0)
        roi_mov = ROI(roi.center + mov_center, window_sz)
        roi_mov.uncertainty = roi.uncertainty
        for i, static_roi in enumerate(rois):
            try:
                area_rel = area(static_roi, roi_mov)/np.prod(roi.window_sz)
                uncert_prop[i, :] += (area_rel * roi_mov.uncertainty, area_rel)
            except:
                pass

    uncert_prop = normalize_prop(uncert_prop)
    return uncert_prop


def main():
    window_sz = (300, 300)
    flow = np.zeros((1080, 1920, 2))
    flow[150, 150, :] = (600, 600)

    ncols = np.floor(float(flow.shape[1]) / window_sz[0]).astype(int)
    nrows = np.floor(float(flow.shape[0]) / window_sz[1]).astype(int)

    rois = create_rois(ncols, nrows, window_sz)
    rois[0].uncertainty = 1

    uncert_prop = np.zeros((len(rois), 2))

    uncert_prop = prop_uncertainty(rois, flow, window_sz, uncert_prop)

    for i, roi in enumerate(rois):
        roi.update_uncertainty(uncert_prop[i, 0])


if __name__ == "__main__":
    main()
