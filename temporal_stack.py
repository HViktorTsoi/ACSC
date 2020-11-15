import os
import time

from mayavi import mlab
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from calibration import crop_ROI_pc

ROI = (0, 1.8, 0.4, 1, -0.5, 0.5)
VOXEL_SIZE = 0.005
# ROI = (1.75, 2, 0.1, 0.45, -0.5, 0.5)
# VOXEL_SIZE = 0.005

# 构建体素
voxel_dimension = np.int_(np.array([ROI[1] - ROI[0], ROI[3] - ROI[2], ROI[5] - ROI[4]]) / VOXEL_SIZE)
voxel_dimension = np.hstack([voxel_dimension, [2]])
voxel_hist = np.zeros(voxel_dimension)
print(voxel_hist.shape)

total_seq = 100

pc_list = []
root = './data/legacy/pointclouds/livox_seq/'
for pc_file in sorted(os.listdir(root)[:total_seq]):
    pc_file = os.path.join(root, pc_file)
    pc = np.load(pc_file)

    # crop
    pc = crop_ROI_pc(pc, ROI)
    #
    # calc voxel coord
    pc = np.int_((pc - [ROI[0], ROI[2], ROI[4], 0]) / VOXEL_SIZE)

    # distribute to voxel
    voxel_hist[pc[:, 0], pc[:, 1], pc[:, 2], 0] += 1
    voxel_hist[pc[:, 0], pc[:, 1], pc[:, 2], 1] += pc[:, 3]
    pc_list.append(pc)

voxel_hist = voxel_hist
plt.hist(voxel_hist[np.where(voxel_hist > 1)], bins=10)
plt.show()

# get density
xx, yy, zz = np.where(voxel_hist[..., 0] > 1)
print(len(xx))

mlab.points3d(xx, yy, zz, voxel_hist[xx, yy, zz, 1] / voxel_hist[xx, yy, zz, 0], mode='point')
# mlab.points3d(xx, yy, zz, color=(1, 0, 0), mode='point')
# mlab.points3d(xx, yy, zz, voxel_hist[xx, yy, zz, 0], mode='point')
# mlab.points3d(xx, yy, zz, mode='cube', scale_factor=1)
# mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(1, 0, 0), mode='point')
mlab.axes()
mlab.show()

pc = np.row_stack(pc_list)
# mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point')
mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], mode='point')
mlab.axes()
mlab.show()
