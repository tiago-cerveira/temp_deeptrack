import numpy as np
from matplotlib import pyplot as plt
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal


class Img:

    def __init__(self, shape):
        self.shape = shape

img = Img((1920, 1080))

hits = []
hits.append([200, 200, 120, 60])
hits.append([800, 800, 80, 200])


def get_roi(img, hits):
    mean = []
    cov = []
    for i in xrange(len(hits)):
        mean.append(np.array([hits[i][0] + hits[i][2] / 2, hits[i][1] + hits[i][3] / 2]))
        # cov.append(np.array([[1000 * hits[i][2], 0], [0, 1000 * hits[i][3]]]))
        cov.append(np.array([[hits[i][2] * 1000, 0], [0, hits[i][3] * 1000]]))

    roi_center = np.random.multivariate_normal(mean[0], cov[0], 1000).astype(np.int)

    return roi_center

roi_centers =  get_roi(img, hits)
print roi_centers
plt.plot(roi_centers[:, 0], roi_centers[:, 1], 'ro')
plt.xlim(0, 1920)
plt.ylim(0, 1080)
plt.show()

"""
multivariate normal
"""
# mean1 = np.array([200, 200])
# cov1 = np.array([[100000, 0], [0, 100000]])
#
# mean2 = np.array([800, 800])
# cov2 = np.array([[100000, 0], [0, 100000]])
#
# # print mean.shape, cov.shape
#
# gaussian1 = np.random.multivariate_normal(mean1, cov1, 1000)
# gaussian2 = np.random.multivariate_normal(mean2, cov2, 1000)
#
# # print gaussian.shape
#
# plt.plot(gaussian1[:, 0], gaussian1[:, 1], 'ro')
# plt.plot(gaussian2[:, 0], gaussian2[:, 1], 'ro')
# plt.xlim(0, 1920)
# plt.ylim(0, 1080)
# plt.show()

"""
to plot stuff
http://matplotlib.org/examples/images_contours_and_fields/contourf_log.html
"""

# N = 10000
# x = np.linspace(0.0, 1920.0, N)
# y = np.linspace(0.0, 1080.0, N)
#
# X, Y = np.meshgrid(x, y)
#
# # A low hump with a spike coming out of the top right.
# # Needs to have z/colour axis on a log scale so we see both hump and spike.
# # linear scale only shows the spike.
# z = (0.5 * bivariate_normal(X, Y, 500.0, 500.0, 200.0, 200.0, 0.0)
#      + 0.5 * bivariate_normal(X, Y, 500.0, 500.0, 1500.0, 1000.0, 0.0))
#
#
#
# # The following is not strictly essential, but it will eliminate
# # a warning.  Comment it out to see the warning.
# # z = ma.masked_where(z <= 0, z)
#
#
# # Automatic selection of levels works; setting the
# # log locator tells contourf to use a log scale:
# fig, ax = plt.subplots()
# cs = ax.contourf(X, Y, z, cmap=cm.PuBu_r)
#
# # Alternatively, you can manually set the levels
# # and the norm:
# #lev_exp = np.arange(np.floor(np.log10(z.min())-1),
# #                    np.ceil(np.log10(z.max())+1))
# #levs = np.power(10, lev_exp)
# #cs = P.contourf(X, Y, z, levs, norm=colors.LogNorm())
#
# # The 'extend' kwarg does not work yet with a log scale.
#
# cbar = fig.colorbar(cs)
#
# plt.show()