# coding=utf-8
from __future__ import print_function, division, absolute_import
import os
import pickle
import sys

import calibration
import numpy as np
import numpy.linalg as LA
import cv2
import yaml
import matplotlib.pyplot as plt

from scipy import optimize

configs = yaml.load(open('configs/config.yaml', 'r').read(), Loader=yaml.FullLoader)
# chess board parameter
# w: number of corners along the long-side
# h: number of corners along the short-side
# grid_size: chessboard grid size(in meters)
# W, H, GRID_SIZE = 9, 7, 0.0365
W = configs['data']['chessboard']['W']
H = configs['data']['chessboard']['H']
GRID_SIZE = configs['data']['chessboard']['GRID_SIZE']


def re_projection_loss(intrinsic_estimate, distortion, corners_world, corners_image, rvec, tvec, vec_RMSE=False):
    # get intrinsic
    intrinsic_estimate_matrix = np.eye(3, dtype=np.float32)
    intrinsic_estimate_matrix[[(0, 1, 0, 1,), (0, 1, 2, 2,)]] = intrinsic_estimate

    # reprojection
    re_projection, _ = cv2.projectPoints(corners_world[:, :3].astype(np.float32),
                                         rvec, tvec, intrinsic_estimate_matrix, distortion)
    # calc RMSE
    if not vec_RMSE:
        RMSE = LA.norm(re_projection.reshape(-1) - corners_image.reshape(-1), ord=2)
        # print('RMSE={:.10f}'.format(RMSE))
    else:
        RMSE = LA.norm(np.squeeze(re_projection) - corners_image, axis=1, ord=2)
    return RMSE


def solve_intrinsic_backward():
    # root = './data/AVT_horizon_1594116694.05'
    root = './data/20200709_two_livox/1594299937.47_0000'

    # load intrinsic and extrinsic parameters
    initial_intrinsic_matrix = np.loadtxt(os.path.join(root, 'intrinsic'))
    distortion = np.loadtxt(os.path.join(root, 'distortion'))
    extrinsic_matrix = np.loadtxt(os.path.join(root, 'parameter/extrinsic'))
    tvec = extrinsic_matrix[:3, 3].reshape(-1, 1)
    rvec, _ = cv2.Rodrigues(extrinsic_matrix[:3, :3])
    # calc re-projection error
    corners_world_list, corners_image_list = pickle.load(open(os.path.join(root, 'parameter/corners.pkl'), 'rb'))

    # concat points
    corners_world_list = np.row_stack(corners_world_list)
    corners_image_list = np.row_stack(corners_image_list)
    assert len(corners_world_list) == len(corners_image_list)

    # initial_intrinsic_matrix = initial_intrinsic_matrix[[(0, 1, 0, 1,), (0, 1, 2, 2,)]]
    initial_intrinsic_matrix = np.zeros(4)
    rst = optimize.minimize(
        re_projection_loss,
        x0=initial_intrinsic_matrix,
        args=(distortion, corners_world_list, corners_image_list, rvec, tvec),
        method='Powell', tol=1e-10, options={"maxiter": 10000000}
    )

    print(len(corners_world_list), rst)

    residual = re_projection_loss(rst.x, distortion, corners_world_list, corners_image_list, rvec, tvec, vec_RMSE=True)
    plt.hist(residual, bins=50)
    plt.show()

    # # keep inliers
    # keep_indices = np.where(residual < 3)
    # corners_world_list = corners_world_list[keep_indices]
    # corners_image_list = corners_image_list[keep_indices]


def calib_camera_intrinsic_matrix(img_list):
    img_corner_list = []  # 在图像平面的二维点
    world_corner_list = []  # 在世界坐标系中的三维点

    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    # 这里是常量，每个图像中都相同
    world_corners = np.zeros((W * H, 3), np.float32)
    world_corners[:, :2] = np.mgrid[0:W, 0:H].T.reshape(-1, 2) * GRID_SIZE

    # 检测角点
    for img in img_list:
        # # 锐化
        # kernel = np.array([
        #     [-1, -1, -1],
        #     [-1, 9, -1],
        #     [-1, -1, -1]],
        #     np.float32)  # 锐化
        # img = cv2.filter2D(img, -1, kernel=kernel)

        # detect corners
        corners = calibration.camera_chessboard_corner_detection(img)
        if corners is not None:
            print('Corners found.')
            # 储存棋盘格角点的世界坐标和图像坐标对
            img_corner_list.append(corners)
            world_corner_list.append(world_corners)
        else:
            print('Error: 未检测到棋盘格！', file=sys.stderr)
            cv2.imshow('ERROR', img)
            cv2.waitKey(0)

    # 标定
    ret, intrinsic_matrix, distortion, rvecs, tvecs = \
        cv2.calibrateCamera(world_corner_list, img_corner_list, img.shape[:2][::-1], None, None)
    if not ret:
        raise Exception('Error: 未找到内参矩阵！')
    return intrinsic_matrix, distortion


def vis_corner_detection():
    W, H = 9, 7
    cap = cv2.VideoCapture(0)
    while True:
        # get a frame
        ret, img = cap.read()

        # detect corners
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # show a frame
        ret, corners = cv2.findChessboardCornersSB(gray, (W, H), None)
        if ret:
            img = cv2.drawChessboardCorners(img, (W, H), corners, ret)
        cv2.imshow("img-corners", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    exit(0)


if __name__ == '__main__':
    # vis_corner_detection()
    solve_intrinsic_backward()
    exit(0)

    # calibration.DEBUG = 2
    #
    # # 内参标定
    # root = './data/1593784979.12/images'
    # img_list = [cv2.imread(file_name) for file_name in [os.path.join(root, file) for file in sorted(os.listdir(root))]]
    # intrinsic_matrix, distortion = calib_camera_intrinsic_matrix(img_list)
    #
    # print('intrinsic matrix:', intrinsic_matrix)
    # print('distortion:', distortion)
