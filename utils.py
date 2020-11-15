# coding=utf-8
from __future__ import print_function, division, absolute_import
import os
import pickle

import sys

import pcl

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
from sklearn.mixture import GaussianMixture
import numpy as np
import numpy.linalg as LA
import transforms3d


class BoxLabeler:
    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        handler = param[0]
        # 左键按下：开始画图
        if event == cv2.EVENT_LBUTTONDBLCLK:
            handler.on_double_click(x, y)
        # 鼠标移动，画图
        elif event == cv2.EVENT_MOUSEMOVE:
            handler.on_mouse_move(x, y)

    def __init__(self, name, img):
        self._drawing = False
        self._start = (-1, -1)
        self._end = (-1, -1)
        self._marker_layer = np.zeros_like(img)
        self._img = img
        self._name = name

        self.BB = None

        # GUI
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, self.mouse_callback, param=(self,))

    def on_double_click(self, x, y):
        """
        处理鼠标双击
        :param x:
        :param y:
        :return:
        """
        self._drawing = not self._drawing
        # 开始绘制
        if self._drawing:
            self._start = (x, y)
            self._end = (-1, -1)
            self.BB = None
        # 停止绘制
        else:
            self._end = (x, y)
            cv2.rectangle(self._marker_layer, self._start, self._end, (0, 255, 0), 1)
            # 获取bb结果
            self.BB = (self._start[0], self._start[1], self._end[0], self._end[1])
            print(self._name, self.BB)

    def on_mouse_move(self, x, y):
        """
        处理鼠标移动
        :param x:
        :param y:
        :return:
        """
        if self._drawing:
            # 清空buffer
            self._marker_layer[...] = 0
            cv2.rectangle(self._marker_layer, self._start, (x, y), (0, 255, 0), 1)

    def render(self):
        """
        渲染
        :return:
        """
        img_vis = cv2.bitwise_or(self._img, self._marker_layer)
        cv2.imshow(self._name, img_vis)


def load_data_pair(root, load_rois):
    # 获取点云和图像对
    pc_root = os.path.join(root, 'pcds')
    img_root = os.path.join(root, 'images')
    pc_file_list = list(sorted(os.listdir(pc_root)))
    img_file_list = list(sorted(os.listdir(img_root)))
    assert len(pc_file_list) == len(img_file_list)

    if load_rois:
        ROI_root = os.path.join(root, 'ROIs')
        roi_file_list = list(sorted(os.listdir(ROI_root)))
    pointcloud_image_pair_list = [
        (
            idx,
            (
                os.path.join(pc_root, pc_file_list[idx]),
                os.path.join(img_root, img_file_list[idx]),
                os.path.join(ROI_root, roi_file_list[idx]) if load_rois else None,
            )
        )
        for idx in range(len(pc_file_list))
    ]
    return pointcloud_image_pair_list


def fit_intensity_pivot(pc, debug):
    intensity = pc[:, 3]
    # gmm = GaussianMixture(n_components=3, covariance_type="diag",
    #                       max_iter=10000, means_init=np.array([[5], [60], [120]])).fit(intensity.reshape(-1, 1))
    gmm = GaussianMixture(n_components=2, covariance_type="diag",
                          max_iter=10000, means_init=np.array([[5], [60]])).fit(intensity.reshape(-1, 1))
    pivot = gmm.means_[:2].mean()
    print('Intensity pivot: {}'.format(pivot))
    if debug > 2:
        import matplotlib.pyplot as plt
        plt.hist(intensity, bins=100)
        plt.show()
    return pivot


def voxelize(pc, voxel_size):
    cloud = pcl.PointCloud_PointXYZI()
    cloud.from_array(pc.astype(np.float32))
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
    return sor.filter().to_array()


def calc_reprojection_error(root, extrinsic_matrix, intrinsic_matrix, distortion, visualize=False):
    """
    calc re-projection error
    :param root:
    :param extrinsic_matrix:
    :param intrinsic_matrix:
    :param distortion:
    :param visualize:
    :return:
    """
    # load data
    pointcloud_image_pair_list = load_data_pair(root, load_rois=False)
    corners_world_list, corners_image_list = pickle.load(open(os.path.join(root, 'parameter/corners.pkl'), 'rb'))
    assert len(corners_world_list) == len(corners_image_list)

    tvec = extrinsic_matrix[:3, 3].reshape(-1, 1)
    rvec, _ = cv2.Rodrigues(extrinsic_matrix[:3, :3])
    RMSE_list = []
    distance_list = []
    for idx in range(len(corners_world_list)):
        corners_world = corners_world_list[idx]
        corners_image = corners_image_list[idx]

        if corners_image is None or corners_world is None:
            continue

        re_projection, _ = cv2.projectPoints(corners_world[:, :3].astype(np.float32),
                                             rvec, tvec, intrinsic_matrix, distortion)
        # calc rmse
        # RMSE = LA.norm(re_projection.reshape(-1) - corners_image.reshape(-1), ord=2)
        RMSE = np.abs(np.squeeze(re_projection) - np.squeeze(corners_image))
        # apply distance weight
        distance = LA.norm(corners_world[:, :3], axis=1, ord=2)
        # RMSE = RMSE * (distance / distance.max()).reshape(-1, 1)
        # RMSE = np.mean(RMSE)

        RMSE_list.append(RMSE)
        distance_list.append(distance)
        print('idx={}, NPE={:.10f}'.format(idx, np.mean(RMSE)))

        # img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
        if visualize:
            corners_reproj = np.squeeze(re_projection)
            # draw re-projection
            img = cv2.imread(pointcloud_image_pair_list[idx][1][1])
            for point in corners_reproj:
                cv2.circle(img, center=(point[0], point[1]), radius=1, color=(0, 0, 255), thickness=2)

            for corner_id in range(len(corners_reproj)):
                cv2.putText(img, '{}'.format(corner_id), tuple(corners_reproj[corner_id] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            cv2.imshow('Re-Projection', img)
            cv2.waitKey(0)

    num_keep = len(RMSE_list)
    RMSE_list = np.vstack(RMSE_list)
    distance_list = np.hstack(distance_list).reshape(-1, 1)
    RMSE_norm = RMSE_list * (distance_list / distance_list.max())
    # print('\n\nAVG NORM RSME: {}'.format(np.mean(RMSE_norm)))
    print('Success: {} / {}'.format(num_keep, len(corners_world_list)))
    return RMSE_norm, np.mean(RMSE_norm)


def any_LiDAR_to_ring(pc, num_beams=32, ring_height=8e-4):
    """
    convert any type of LiDAR point cloud to ring-based LiDAR style
    :param pc: input point cloud, shape of Nx4(x,y,z,intensity)
    :param num_beams: number of beams
    :param ring_height: the "line width" of a ring
    :return: ring-stype point cloud, shape of Nx5(x,y,z,intensity, ring ID)
    """
    pitch = np.arctan(pc[:, 2] / (LA.norm(pc[:, :2], axis=1, ord=2) + 1e-10))
    pitch = np.nan_to_num(pitch)

    beams = np.linspace(pitch.min(), pitch.max(), num=num_beams + 1)

    rings = []
    for beam_id, beam_angle in enumerate(beams):
        ring = pc[np.where(
            (pitch > beam_angle) &
            (pitch < beam_angle + ring_height)
        )]
        ring_ids = beam_id * np.ones_like(ring[:, 0])

        # sort according to azimuth
        az = np.arctan(ring[:, 1] / ring[:, 0])
        ring = ring[np.argsort(az), :]

        rings.append(np.column_stack([ring, ring_ids]))
    ring_pc = np.row_stack(rings)
    return ring_pc


def visualize(pc, color=None, show=False, mode='points'):
    try:
        from mayavi import mlab
    except ImportError:
        print('mayavi not found, skip visualize')
        return

    mlab.figure('pc', bgcolor=(0.05, 0.05, 0.05))
    if color is None:
        if mode == 'points':
            out = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], -pc[:, 3], mode='point')
        elif mode == 'cube':
            out = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], -pc[:, 3],
                                mode='cube', scale_mode='none', scale_factor=0.003)

    else:
        if mode == 'points':
            out = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=color,
                                scale_factor=0.04)
        elif mode == 'cube':
            out = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=color,
                                mode='cube', scale_mode='none', scale_factor=0.004)

    # mlab.axes()
    if show:
        out.actor.property.lighting = False
        mlab.show()


def visualize_text(point, text):
    try:
        from mayavi import mlab
    except ImportError:
        return
    mlab.figure('pc', bgcolor=(0.05, 0.05, 0.05))
    mlab.text3d(point[0], point[1], point[2] + 0.03, '{}'.format(int(point[3])), scale=0.02, color=(0, 0.8, 0))
    # mlab.text3d(point[0], point[1], point[2] + 0.03, '{}'.format(int(point[3])), scale=0.02, color=(1, 1, 1))


def visualize_colored_pointcloud(pc):
    try:
        from mayavi import mlab
    except ImportError:
        print('mayavi not found, skip visualize')
        return
        # plot rgba points
    mlab.figure('pc', bgcolor=(0.05, 0.05, 0.05))
    # 构建lut 将RGB颜色索引到点
    lut_idx = np.arange(len(pc))
    lut = np.column_stack([pc[:, 4:][:, ::-1], np.ones_like(pc[:, 0]) * 255])
    # plot
    p3d = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], lut_idx, mode='point')
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut_idx)
    p3d.module_manager.scalar_lut_manager.lut.table = lut
    # mlab.axes()
    mlab.show()


def pose_to_matrix(pose):
    assert len(pose) == 6
    tvec, rvec = pose[:3].reshape(-1, 1), pose[3:]
    rotation_m = transforms3d.euler.euler2mat(rvec[0], rvec[1], rvec[2])
    # # 罗德里格斯变换, 将旋转角转换为旋转矩阵
    # rotation_m, _ = cv2.Rodrigues(rvec)

    # 拼接最终的外参
    extrinsic_matrix = np.hstack([rotation_m, tvec])
    extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])

    return extrinsic_matrix
