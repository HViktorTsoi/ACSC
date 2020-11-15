# coding=utf-8
from __future__ import print_function, division, absolute_import

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy.linalg as LA
import yaml
from scipy.interpolate import griddata

import utils

import cv2
import numpy as np

SCALE_FACTOR = 2


def undistort_projection(points, intrinsic_matrix, extrinsic_matrix):
    points = np.column_stack([points, np.ones_like(points[:, 0])])

    # 外参矩阵
    points = np.matmul(extrinsic_matrix, points.T, )

    # 内参矩阵
    points = np.matmul(intrinsic_matrix, points[:3, :], ).T

    # 深度归一化
    points[:, :2] /= points[:, 2].reshape(-1, 1)

    return points


def back_projection(points, intrinsic_matrix, extrinsic_matrix):
    # 还原深度
    points[:, :2] *= points[:, 2].reshape(-1, 1)

    # 还原相平面相机坐标
    points[:, :3] = np.matmul(LA.inv(intrinsic_matrix), points[:, :3].T).T

    # 还原世界坐标
    # 旋转平移矩阵
    R, T = extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3]
    points[:, :3] = np.matmul(LA.inv(R), points[:, :3].T - T.reshape(-1, 1)).T

    return points


def img_to_pc(pc, img, extrinsic_matrix, intrinsic_matrix, distortion):
    # project pointcloud to image
    projection_points = undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)
    projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])

    # crop
    projection_points = projection_points[np.where(
        (projection_points[:, 0] > 0) &
        (projection_points[:, 0] < img.shape[1]) &
        (projection_points[:, 1] > 0) &
        (projection_points[:, 1] < img.shape[0])
    )]

    # depth map projection
    depth_map = np.zeros_like(img, dtype=np.float32)
    depth_map[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 0] = projection_points[:, 2]
    depth_map[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 1] = projection_points[:, 3]

    available_depth_indices = np.where(depth_map[..., 0] > 0)
    projection_points = np.row_stack([available_depth_indices[1], available_depth_indices[0],
                                      depth_map[available_depth_indices][..., 0], depth_map[available_depth_indices][..., 1]]).T

    # 图像点云深度匹配
    RGB = img[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), :]
    print(RGB)

    projection_points = np.column_stack([projection_points, RGB])
    print(projection_points)

    # back projection
    pc = back_projection(projection_points, intrinsic_matrix, extrinsic_matrix)

    utils.visualize_colored_pointcloud(pc)
    return pc


def pc_to_img(pc, img, extrinsic_matrix, intrinsic_matrix, distortion):
    # # 投影验证
    # projection_points, jav = cv2.projectPoints(pc[:, :3].astype(np.float32),
    #                                            rvec, tvec, intrinsic_matrix, distortion)
    # pc = pc[:300000, :]
    # pc = utils.voxelize(pc, voxel_size=0.04)
    # 投影验证
    projection_points = undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)

    # 裁切到图像平面
    projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])
    projection_points = projection_points[np.where(
        (projection_points[:, 0] > 0) &
        (projection_points[:, 0] < img.shape[1]) &
        (projection_points[:, 1] > 0) &
        (projection_points[:, 1] < img.shape[0])
    )]

    # scale
    img = cv2.resize(img, (int(img.shape[1] / SCALE_FACTOR) + 1, int(img.shape[0] / SCALE_FACTOR) + 1))
    projection_points[:, :2] /= SCALE_FACTOR

    board = np.zeros_like(img)

    # 提取边缘
    edge = np.uint8(np.absolute(cv2.Laplacian(
        cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F)))
    # board[..., 0] = board[..., 1] = edge ** 1.5
    board[...] = img[..., ::-1]

    # # colors = plt.get_cmap('gist_ncar_r')(projection_points[:, 3] / 255) * 2
    # colors = plt.get_cmap('gist_ncar_r')((30 - projection_points[:, 2]) / 30)
    # img_size = img.shape[:2][::-1]
    # grid_x, grid_y = np.mgrid[0:img_size[0]:1, 0:img_size[1]:1]
    # chs = [griddata(projection_points[:, 0:2], colors[:, 2 - idx], (grid_x, grid_y), method='linear').T for idx in range(3)]
    # board = np.stack(chs, axis=-1)

    # 反射率可视化
    # colors = plt.get_cmap('gist_ncar_r')(projection_points[:, 3] / 255) ** 2
    colors = plt.get_cmap('gist_ncar_r')(projection_points[:, 3] / 255) ** 2
    # colors = plt.get_cmap('gist_ncar_r')((30 - projection_points[:, 2]) / 30)
    for idx in range(3):
        board[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 2 - idx] = colors[:, idx] * 255

    # board = board[120:, ...]

    # board = cv2.resize(board, dsize=(board.shape[1] // 2, board.shape[0] // 2))

    cv2.imshow('Projection', board)
    cv2.waitKey(0)

    cv2.imwrite('/tmp/board.png', board)

    return board


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help="config file path")
    args = parser.parse_args()

    configs = yaml.load(open(args.config, 'r').read(), Loader=yaml.FullLoader)

    # load data
    root = configs['data']['root']

    # load validation data pair
    pointcloud_image_pair_list = utils.load_data_pair(root, load_rois=False)

    # load intrinsic and extrinsic parameters
    intrinsic_matrix = np.loadtxt(os.path.join(root, 'intrinsic'))
    distortion = np.loadtxt(os.path.join(root, 'distortion'))
    extrinsic_matrix = np.loadtxt(os.path.join(root, 'parameter/extrinsic'))
    tvec = extrinsic_matrix[:3, 3].reshape(-1, 1)
    rvec, _ = cv2.Rodrigues(extrinsic_matrix[:3, :3])

    # for frame_id in range(1, 13):
    for idx, (pc_file, img_file, ROI_file) in pointcloud_image_pair_list:
        # frame_id = 5
        pc = np.load(pc_file)
        img = cv2.imread(img_file)

        # 消除图像distortion
        img = cv2.undistort(img, intrinsic_matrix, distortion)

        # img = img[..., ::-1]
        # process pc
        pc = pc[np.where(
            (pc[:, 0] > 0) &
            (abs(pc[:, 0]) < 100) &
            (pc[:, 2] > -3)
        )]
        # reprojection
        pc_to_img(pc, img, extrinsic_matrix, intrinsic_matrix, distortion)
        # colorize
        img_to_pc(pc, img, extrinsic_matrix, intrinsic_matrix, distortion)

    # calc reprojection error
    utils.calc_reprojection_error(root, extrinsic_matrix, intrinsic_matrix, distortion, visualize=True)
