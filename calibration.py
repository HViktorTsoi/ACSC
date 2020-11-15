# coding=utf-8
from __future__ import print_function, division, absolute_import

import json
import multiprocessing
import pickle
import sys
import yaml
import argparse

import utils
import array
import os
import time

import numpy as np
import numpy.linalg as LA
import transforms3d
from sklearn.decomposition import PCA
import pcl
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy import signal, stats

import cv2
import segmentation_ext

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, type=str, help="config file path")
args = parser.parse_args()

configs = yaml.load(open(args.config, 'r').read(), Loader=yaml.FullLoader)
configs_data_collection = yaml.load(open('./configs/data_collection.yaml', 'r').read(), Loader=yaml.FullLoader)

# whether use automic chessboard localization
AUTOMATIC_CHESSBOARD_LOCALIZATION = configs_data_collection['AUTOMATIC_CHESSBOARD_LOCALIZATION']

# if there is pre-labeled datadensity
RELABEL = configs_data_collection['RELABEL']

# scene size when labeling chessboards
BOUND = configs_data_collection['BOUND']  # xmin,xmax,ymin,ymax,zmin,zmax

# resolution of displayed point cloud projection, adjust it according to your screen size
VIRTUAL_CAM_INTRINSIC = configs_data_collection['VIRTUAL_CAM_INTRINSIC']  # 虚拟相机内参

# long-side resolution of rendered image for labeling chessboard location
RENDER_IMG_WIDTH = 500

# chess board parameter
# w: number of corners along the long-side
# h: number of corners along the short-side
# grid_size: chessboard grid size(in meters)
# W, H, GRID_SIZE = 9, 7, 0.0365
W = configs['data']['chessboard']['W']
H = configs['data']['chessboard']['H']
GRID_SIZE = configs['data']['chessboard']['GRID_SIZE']

# threshhold for 3D corner detection loss
OPTIMIZE_TH = configs['calibration']['OPTIMIZE_TH']

# patch size used to calc resample density
RESAMPLE_PATCH_SIZE = configs['calibration']['RESAMPLE_PATCH_SIZE']

# desired density after re-sample, points / cm^2
RESAMPLE_DENSITY = configs['calibration']['RESAMPLE_DENSITY']

# the margin reserved after segmentation from ROI pillar, preventing cut to much of chessboard
HEIGHT_SEGMENTATION_MARGIN = configs['calibration']['HEIGHT_SEGMENTATION_MARGIN']

# number of bins for calc z-axis variance, the more sparse point cloud is, the lower bins should be
HEIGHT_SEGMENTATION_BINS = configs['calibration']['HEIGHT_SEGMENTATION_BINS']

# num of workers to process the frames
NUM_WORKERS = configs['calibration']['NUM_WORKERS']

# 0: No Vis
# 1: Vis 3D detection result
# 2: Vis 3D detection result，intensity distribution，camera calibration
DEBUG = configs['calibration']['DEBUG']

# load data
root = configs['data']['root']


def load_pc(bin_file_path):
    """
    load pointcloud file (KITTI format)
    :param bin_file_path:
    :return:
    """
    with open(bin_file_path, 'rb') as bin_file:
        pc = array.array('f')
        pc.frombytes(bin_file.read())
        pc = np.array(pc).reshape(-1, 4)
        return pc


def transfer_by_normal_vector(pc, norm):
    #
    norm_from = norm
    norm_to = np.array([0, 0, -1])

    # rotate angle
    theta = np.arccos(np.dot(norm_from, norm_to) / (LA.norm(norm_from) * LA.norm(norm_to)))

    # rotate pivot
    # a1, a2, a3 = norm_to
    # b1, b2, b3 = norm_from
    a1, a2, a3 = norm_from
    b1, b2, b3 = norm_to
    pivot = np.array([
        a2 * b3 - a3 * b2,
        a3 * b1 - a1 * b3,
        a1 * b2 - a2 * b1
    ])

    # convert to SO3 rotate mat
    SO3 = transforms3d.axangles.axangle2mat(pivot, theta)

    # apply transform
    rotated_pc = np.dot(pc, SO3.T)
    return SO3, rotated_pc


def transfer_by_pca(pc):
    """
    transfer chessboard to xOy plane
    REF: https://github.com/mfxox/ILCC/tree/master/ILCC
    :param pc:
    :return:
    """
    # to rotate the arr correctly, the direction of the axis has to be found correct(PCA also shows the axis but not the direction of axis)
    #
    pca = PCA(n_components=3)
    pca.fit(pc)

    ####################################################
    # there are C(6,3) possible axis combination, so the following constraint should be applied:
    # there are there requirements for the coordinates axes for the coordinate system of the chessboard
    # 1. right-hand rule
    # 2. z axis should point to the origin
    # 3. the angle between y axis of chessboard and z axis of LiDAR point cloud less than 90 deg
    ####################################################
    trans_mat = pca.components_
    # switch x and y axis
    trans_mat[[0, 1]] = trans_mat[[1, 0]]
    # cal z axis to obey the right hands
    trans_mat[2] = np.cross(trans_mat[0], trans_mat[1])

    # to make angle between y axis of chessboard and z axis of LiDAR point cloud less than 90 deg
    sign2 = np.sign(np.dot(trans_mat[1], np.array([0, 0, 1])))
    # print "sign2", sign2
    # we only need the property that Y-axis to be transformed, but x-axis must be transformed together to keep right-hand property
    trans_mat[[0, 1]] = sign2 * trans_mat[[0, 1]]

    # to make the norm vector point to the side where the origin exists
    # the angle  between z axis and the vector  from one point on the board to the origin should  be less than 90 deg
    sign = np.sign(np.dot(trans_mat[2], 0 - pc.mean(axis=0).T))

    # print "sign", sign
    # need Z-axis to be transformed, we transform X-axis along with it just to keep previous property
    trans_mat[[0, 2]] = sign * trans_mat[[0, 2]]

    transfered_pc = np.dot(pc, trans_mat.T)
    # print pca.components_
    # print "x,y,cross", np.cross(pca.components_[1], pca.components_[2])

    return trans_mat, transfered_pc


def generate_chessboard_corner():
    # 生成角点

    xx, yy = np.meshgrid(np.arange(1, H + 1) * GRID_SIZE, np.arange(1, W + 1) * GRID_SIZE)
    xx, yy = xx.reshape(-1), yy.reshape(-1)
    corner = np.column_stack((xx, yy, np.zeros_like(yy)))

    # 生成索引 cv2中棋盘格是左下角为起始，按照高度方向递增
    corner_order = (yy / GRID_SIZE - 1) + W * (xx / GRID_SIZE - 1)
    corner = np.column_stack([corner, corner_order])
    # 按照棋盘格特定模式对corner进行排序 左下角是起点，按照y轴(棋盘格长边)递增
    corner = corner[np.argsort(corner_order), :]

    # 偏移到以棋盘中点为原点
    corner[:, :2] -= np.array([(H + 1) / 2, (W + 1) / 2]) * GRID_SIZE

    # 返回结果
    return corner


def fast_polygon_test(pc, bound):
    """
    test whether point in polygon(corners is bound)
    Notice: because the the border of ideal chessboard that we aim to solve is just parallel to x and y axis,
    the polygon test can be simplified extremely
    :param pc:
    :param bound:
    :return:
    """
    not_in_polygon = (pc[:, 0] < min(bound[:, 0])) | (pc[:, 0] > max(bound[:, 0])) \
                     | (pc[:, 1] < min(bound[:, 1])) | (pc[:, 1] > max(bound[:, 1]))
    return not_in_polygon


def matching_loss(theta_t, pointcloud, intensity_pivot, x_res, y_res, grid_len):
    """
    solving the relative position between real chessboard point cloud and ideal chessboard model
    :param theta_t: the parameter to optimize
    :param pointcloud:
    :param intensity_pivot:
    :param x_res:
    :param y_res:
    :param grid_len:
    :return:
    """

    # 这里要使用新的array，否则引用pointcloud会在每次优化的过程中都改变pointcloud的值
    trans_pointcloud = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], theta_t[0]),
                              (pointcloud[:, :3] + np.array([[theta_t[1], theta_t[2], 0]])).T).T
    next_pc = np.column_stack([trans_pointcloud, pointcloud[:, 3]])
    bound = np.array([[0, 0], [0, y_res], [x_res, y_res], [x_res, 0]]) * \
            grid_len - np.array([x_res, y_res]) * grid_len / 2

    x_grid_arr = (np.array(range(0, x_res + 1)) - float(x_res) / 2) * grid_len
    y_grid_arr = (np.array(range(0, y_res + 1)) - float(y_res) / 2) * grid_len

    tic = time.time()

    # The vectorized implementation of polygon test and cost calculation, which may not be intuitive to understand
    # for intuitive understanding, you may read the naive implementation

    # which grid each point lies in, according to its position
    point_grid = np.int_((next_pc[:, :2] + [x_res * grid_len / 2, y_res * grid_len / 2]) / grid_len)

    # the ground-truth color that each point supposed to be, according to its grid on the chessboard
    supposed_colors = np.int_((point_grid[:, 0] % 2) != (point_grid[:, 1] % 2))

    # the actual color of each point
    actual_colors = (np.sign(next_pc[:, 3] - intensity_pivot) + 1) / 2

    # compare supposed color and actual colors
    logits_color_unequal = (actual_colors != supposed_colors)
    # res_color_unequal = np.abs(supposed_colors - actual_colors)

    # whether points are in the chessboard square
    logits_not_in_bound = fast_polygon_test(next_pc, bound)

    # decide which point to apply cost
    # two condition should be punished:
    # 1. point is not in chessboard box
    # 2. the point color doesn't match
    apply_cost_mask = np.int_(np.bitwise_or(logits_color_unequal, logits_not_in_bound))
    # apply_cost_mask = np.int_(logits_not_in_bound)

    # the cost are calculated by the distance between each point and its nearest grid
    points_grid_min_dist_x = np.min(np.abs(
        np.tile(next_pc[:, 0], (len(x_grid_arr), 1), ).T - x_grid_arr
    ), axis=1)
    points_grid_min_dist_y = np.min(np.abs(
        np.tile(next_pc[:, 1], (len(y_grid_arr), 1), ).T - y_grid_arr
    ), axis=1)
    cost_vec = points_grid_min_dist_x + points_grid_min_dist_y

    # calc the accumulated cost for each point that satisfies punish condition
    cost = np.sum(cost_vec * apply_cost_mask)
    # cost = np.sum(cost_vec * apply_cost_mask * res_color_unequal)

    toc = time.time()

    return cost


def label_image_marker_interactive_gui(input_img):
    img = np.copy(input_img)
    render_img = cv2.resize(img, (RENDER_IMG_WIDTH, int(img.shape[0] / img.shape[1] * RENDER_IMG_WIDTH)))

    labeler_img = utils.BoxLabeler('image', render_img)
    # 执行渲染
    while True:
        labeler_img.render()
        # 退出
        if cv2.waitKey(1) == 13:
            cv2.destroyAllWindows()
            break

    # 获取渲染图像中BB对应的真实坐标
    if labeler_img.BB is None:
        return img, None

    bb = np.int_(np.array(labeler_img.BB) / RENDER_IMG_WIDTH * img.shape[1])

    # 将不包含棋盘的图像部分屏蔽
    X_MAX = int(max(bb[[0, 2]]))
    X_MIN = int(min(bb[[0, 2]]))
    Y_MAX = int(max(bb[[1, 3]]))
    Y_MIN = int(min(bb[[1, 3]]))

    # cv2.imshow('', img)
    # cv2.waitKey(0)
    return X_MIN, X_MAX, Y_MIN, Y_MAX


def convert_BB_2d3d(BB_bird):
    # 还原到world坐标系
    BB_bird = BB_bird / VIRTUAL_CAM_INTRINSIC
    BB_bird[[0, 2]] += BOUND[2]  # 鸟瞰图宽度方向，对应点云y轴
    BB_bird[[1, 3]] += BOUND[0]  # 鸟瞰图高度方向，对应点云x轴

    # 3D Bounding Box
    X_MIN = min(BB_bird[[1, 3]])
    X_MAX = max(BB_bird[[1, 3]])
    Y_MIN = min(BB_bird[[0, 2]])
    Y_MAX = max(BB_bird[[0, 2]])

    return X_MIN, X_MAX, Y_MIN, Y_MAX


def label_pointcloud_marker_interactive_gui(input_pc):
    pc = np.copy(input_pc)

    # 限制范围
    pc = pc[np.where(
        (BOUND[0] < pc[:, 0]) & (pc[:, 0] < BOUND[1]) &
        (BOUND[2] < pc[:, 1]) & (pc[:, 1] < BOUND[3]) &
        (BOUND[4] < pc[:, 2]) & (pc[:, 2] < BOUND[5])
    )]

    # bird view
    img_size_bird = (int((BOUND[1] - BOUND[0]) * VIRTUAL_CAM_INTRINSIC),
                     int((BOUND[3] - BOUND[2]) * VIRTUAL_CAM_INTRINSIC))  # height, width
    img_bird = np.zeros(img_size_bird + (3,), dtype=np.uint8)

    # 投影
    bird_x = np.int_((pc[:, 1] - BOUND[2]) * VIRTUAL_CAM_INTRINSIC)
    bird_y = np.int_((pc[:, 0] - BOUND[0]) * VIRTUAL_CAM_INTRINSIC)

    img_bird[bird_y, bird_x, 2] = pc[:, 2] * VIRTUAL_CAM_INTRINSIC
    img_bird[bird_y, bird_x, 1] = pc[:, 3] * 10

    # 交互绘图
    labeler_bird = utils.BoxLabeler('bird-view', img_bird)

    # 执行渲染
    while True:
        labeler_bird.render()
        # 退出
        if cv2.waitKey(1) == 13:
            # 保证获取了所有标定
            if labeler_bird.BB is None:
                print('鸟瞰图未标定！！！', file=sys.stderr)
                continue
            cv2.destroyAllWindows()
            break

    if labeler_bird.BB is None:
        raise Exception('必须在鸟瞰图上将标定板的位置框选出来！！！')

    # 获取bounding box
    BB_bird = np.array(labeler_bird.BB)
    print('BB_bird', BB_bird)

    # get labeled ROI
    X_MIN, X_MAX, Y_MIN, Y_MAX = convert_BB_2d3d(BB_bird)

    # use scene max and min height, as the top and bottom of the pillar
    Z_MIN, Z_MAX = BOUND[-2:]

    return X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX


def crop_ROI_img(img, ROI):
    if ROI is None:
        return img
    X_MIN, X_MAX, Y_MIN, Y_MAX = ROI
    img[:, :X_MIN] = img[:, X_MAX:] = 128
    img[:Y_MIN, :] = img[Y_MAX:, :] = 128
    return img


def locate_chessboard(input):
    # limit
    pc = input[np.where(
        (BOUND[0] < input[:, 0]) & (input[:, 0] < BOUND[1]) &
        (BOUND[2] < input[:, 1]) & (input[:, 1] < BOUND[3]) &
        (BOUND[4] < input[:, 2]) & (input[:, 2] < BOUND[5])
    )]
    pc = utils.voxelize(pc, voxel_size=configs['calibration']['RG_VOXEL'])

    # region growing segmentation
    segmentation = segmentation_ext.region_growing_kernel(
        pc,
        configs['calibration']['RG_GROUND_REMOVAL'],
        configs['calibration']['RG_NUM_NEIGHBOR'],
        configs['calibration']['RG_MIN_REGION_SIZE'],
        configs['calibration']['RG_MAX_REGION_SIZE'],
        configs['calibration']['RG_SMOOTH_TH'],
        configs['calibration']['RG_CURV_TH']
    )
    segmentation = segmentation[np.where(segmentation[:, 4] > -1)]

    if DEBUG > 1:
        seg_vis = np.copy(segmentation)
        seg_vis[:, 3] = seg_vis[:, 4] + 10
        utils.visualize(seg_vis, show=True)

    # find in segs that bset fits the chessboard
    std_diag = LA.norm([(W + 1) * GRID_SIZE, (H + 1) * GRID_SIZE], ord=2)
    seg_costs = []
    for label_id in range(int(segmentation[:, 4].max() + 1)):
        seg = segmentation[np.where(segmentation[:, 4] == label_id)]

        # if len(seg) > 1500:
        #     continue

        # remove components that are too big
        diag = LA.norm(np.max(seg[:, :3], axis=0) - np.min(seg[:, :3], axis=0), ord=2)
        if diag > std_diag * 2:
            continue

        # transfer to XOY plane
        rot1_3d, transed_pcd = transfer_by_pca(seg[:, :3])
        trans_3d = transed_pcd.mean(axis=0)
        seg[:, :3] = transed_pcd - trans_3d

        fixed_intensity_pivot = 50
        # 优化实际点云和chessboard之间的变换参数
        rst = minimize(matching_loss, x0=np.zeros(3), args=(seg, fixed_intensity_pivot, H + 1, W + 1, GRID_SIZE,),
                       method='L-BFGS-B', tol=1e-10, options={"maxiter": 10000000})

        seg_costs.append([label_id, rst.fun / len(seg) * 1000])

        if DEBUG > 4:
            print(rst.fun / len(seg) * 1000)
            utils.visualize(seg, show=True, mode='cube')

    if len(seg_costs) == 0:
        return 0, 0, 0, 0, 0, 0
    # find the one with minimal cost
    seg_costs = np.array(seg_costs)
    label_id = seg_costs[np.argmin(seg_costs[:, 1]), 0]
    segmentation = segmentation[np.where(segmentation[:, 4] == label_id)]
    print('\nLocalization done. min cost={}'.format(seg_costs[:, 1].min()))

    X_MAX, Y_MAX, Z_MAX = np.max(segmentation[:, :3], axis=0) + 0.03
    X_MIN, Y_MIN, Z_MIN = np.min(segmentation[:, :3], axis=0) - 0.03
    return X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX


def height_segmentation(pc):
    if len(pc) == 0:
        return pc
    # x-y var along Z-axis
    height_hist, edge, _ = stats.binned_statistic(pc[:, 2], LA.norm(pc[:, :2] - np.mean(pc[:, :2], axis=0), axis=1, ord=1), statistic='std', bins=HEIGHT_SEGMENTATION_BINS)
    # height_hist, edge = np.histogram(pc[:, 2], bins=40)
    height_hist_diff = height_hist[1:] - height_hist[: -1]

    if DEBUG > 2:
        plt.bar(edge[1:], height_hist)
        plt.show()
        plt.plot(height_hist_diff)
        plt.show()

    # find segmentation pivot
    # !!! suppose there is no roof top
    # !! suppose the top border of chess board is the upest points of the pillar
    # sort height hist diff
    sorted_diff_indices = np.argsort(-height_hist_diff)

    # find top-k possible segmentation pivox
    K = 5
    possible_pivots = edge[sorted_diff_indices + 1][:K]

    # if we choose a pivot, calc the estimated chessboard diag length
    chessboard_top = pc[:, 2].max()
    estimated_chessboard_height = chessboard_top - possible_pivots
    gt_chessboard_height = (W + 1) * GRID_SIZE

    # how much the possible pivots are close to real calibration board
    # the height-close and var-diff score are fused with a exponentiation weight
    height_close_factor = 1 - np.abs(estimated_chessboard_height - gt_chessboard_height) / gt_chessboard_height
    pivot_scores = (height_close_factor ** 3) * (height_hist_diff[sorted_diff_indices][:K] ** 1)

    # find a pivox that maximize the combined score
    pivot = np.argmax(pivot_scores)

    Z_MIN = possible_pivots[pivot] - HEIGHT_SEGMENTATION_MARGIN

    # crop Z-axis ROI based on segmentation
    pc = pc[np.where((pc[:, 2] > Z_MIN))]

    return pc


def crop_ROI_pc(pc, ROI):
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX = ROI
    # 裁剪ROI
    pc = pc[np.where(
        (pc[:, 0] > X_MIN) & (pc[:, 0] < X_MAX)
        & (pc[:, 1] > Y_MIN) & (pc[:, 1] < Y_MAX)
        & (pc[:, 2] > Z_MIN) & (pc[:, 2] < Z_MAX)
    )]

    # segment chessboard according to z-axis density
    pc = height_segmentation(pc)

    if DEBUG > 3:
        if len(pc) < 100:
            return pc
        utils.visualize(pc, mode='cube', show=True)
    return pc


def ray_projection(pc, A, B, C, D):
    # 使用lidar的射线模型将点投影到标定板平面上
    x0, y0, z0 = pc[:, 0], pc[:, 1], pc[:, 2]

    # 直线方程就是X=x0t,Y=y0t,Z=z0t,其中x0y0z0就是实际点坐标，因为是过原点的射线
    line_func = lambda t: (x0 * t, y0 * t, z0 * t)

    # 直线和平面交点
    t = -D / (A * x0 + B * y0 + C * z0)
    x, y, z = line_func(t)

    # 获取坐标
    pc[:, :3] = np.column_stack([x, y, z])

    return pc


def ransac_segment(pc):
    cloud = pcl.PointCloud_PointXYZI()
    cloud.from_array(pc.astype(np.float32))
    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.04)
    seg.set_normal_distance_weight(0.001)
    seg.set_max_iterations(100)
    return seg.segment()


def fit_chessboard_plane(pc):
    thresh_list = [0.1, 0.02]
    for iter_time, thresh in enumerate(thresh_list):
        # fit plane
        indices, coefficients = ransac_segment(pc)

        # calc point-to-plane distance
        A, B, C, D = coefficients
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        dist = np.abs(A * x + B * y + C * z + D) / LA.norm([A, B, C], ord=2)
        pc = pc[np.where(dist < thresh)]

    if DEBUG > 2:
        pc = pc[indices, :]
        utils.visualize(pc, show=True)
        # plt.hist(dist, bins=20)
        # plt.show()

    return indices, coefficients


def resample(pc):
    """
    re-sample pc, to ease the un-uniform distribution of points on chessboard
    :param pc:
    :return:
    """
    tic = time.time()
    x_max, y_max = np.max(pc[:, :2], axis=0)
    x_min, y_min = np.min(pc[:, :2], axis=0)

    density_list = []

    indices = []
    for x_bound in np.arange(x_min, x_max, RESAMPLE_PATCH_SIZE):
        for y_bound in np.arange(y_min, y_max, RESAMPLE_PATCH_SIZE):
            # search every patch
            patch_indices = np.where(
                (pc[:, 0] > x_bound) &
                (pc[:, 0] < x_bound + RESAMPLE_PATCH_SIZE) &
                (pc[:, 1] > y_bound) &
                (pc[:, 1] < y_bound + RESAMPLE_PATCH_SIZE)
            )

            # resample those patch that have two many points
            max_sample_points = int(RESAMPLE_DENSITY * (RESAMPLE_PATCH_SIZE * 100) ** 2)
            if len(patch_indices[0]) > max_sample_points:
                indices.append(
                    np.random.choice(patch_indices[0], max_sample_points, replace=False)
                )
            else:
                indices.append(patch_indices[0])

            density_list.append(len(pc[patch_indices]))
    indices = np.hstack(indices)
    pc = pc[indices, :]
    toc = time.time()
    print('Re-sample time: {}s'.format(toc - tic))

    if DEBUG > 2:
        plt.hist(density_list, bins=20)
        plt.show()

    return pc


def feature_refinement(pc):
    """
    chessboard pointcloud refinement
    :param pc:
    :return:
    """
    # 拟合标定板平面
    indices, coefficients = fit_chessboard_plane(pc)
    pc = pc[indices, :]

    # 拟合intensity分布的gap，后续以此为pivot将点云二值化
    intensity_pivot = utils.fit_intensity_pivot(pc, DEBUG)

    # 平面参数归一化
    coefficients = np.array(coefficients) / -coefficients[-1]
    A, B, C, D = coefficients

    # # 沿x轴方向把有噪声的点投影到平面上
    # pc[:, 0] = (1 - B * pc[:, 1] - C * pc[:, 2]) / A

    # 按照射线模型将点投影到平面上
    pc = ray_projection(pc, A, B, C, D)

    if DEBUG > 2:
        utils.visualize(pc, mode='cube', show=True)
    # 将标定板转换到xoy平面上，方便将它和chessboard model进行配准
    # 都在xoy平面上，只需要2d的xy偏移和旋转角就可以配准了
    rot_3d, transed_pcd = transfer_by_pca(pc[:, :3])
    # rot1_3d, transed_pcd = transfer_by_normal_vector(pc[:, :3], norm=np.array([A, B, C]))
    trans_3d = transed_pcd.mean(axis=0)
    pc[:, :3] = transed_pcd - trans_3d

    # uniformally re-sample pointcloud
    pc = resample(pc)

    if DEBUG > 2:
        utils.visualize(pc, mode='cube', show=True)

    # voxelize 减小计算量
    pc = utils.voxelize(pc, voxel_size=0.002)

    print('Points after voxelize: {}'.format(len(pc)))

    return pc, trans_3d, rot_3d, intensity_pivot


def world_corner_post_process(corners, rot_2d, trans_2d, rot_3d, trans_3d):
    # 将corner与原始点云平面对齐
    corners[:, :3] = np.dot(corners[:, :3], LA.inv(rot_2d.T)) - trans_2d
    # 将corner还原到原始点云对应的3D空间
    corners[:, :3] = np.dot(corners[:, :3] + trans_3d, LA.inv(rot_3d.T))

    # 确保索引是从左下方的角点开始的 保证corners的顺序和图像中一致
    if corners[0, 2] > corners[-1, 2]:  # 比较z轴
        # 保证首点在末点下方
        corners = corners[::-1, :]
    elif corners[0, 2] == corners[-1, 2]:
        # 保证首点在末点左侧
        if corners[0, 1] < corners[-1, 1]:  # 比较y轴
            corners = corners[::-1, :]

    return corners


def pointcloud_chessboard_corner_detection(input_pc):
    pc = np.copy(input_pc)

    # refine measurement
    pc, trans_3d, rot_3d, intensity_pivot = feature_refinement(pc)

    # 优化实际点云和chessboard之间的变换参数
    rst = minimize(matching_loss, x0=np.zeros(3), args=(pc, intensity_pivot, H + 1, W + 1, GRID_SIZE,),
                   # method='Powell', tol=1e-10, options={"maxiter": 10000000})
                   method='L-BFGS-B', tol=1e-10, options={"maxiter": 10000000})

    theta, offset_x, offset_y = rst.x
    # theta, offset_x, offset_y = [-0.09495837, -0.00408983, 0.02385411]
    # theta, offset_x, offset_y = [-6.76778002e-03, -5.91509102e-05, 5.63815809e-03]
    rot_2d = transforms3d.axangles.axangle2mat([0, 0, 1], theta)
    trans_2d = np.array([offset_x, offset_y, 0])

    # 生成平面内的规则角点
    corners = generate_chessboard_corner()

    if DEBUG > 0:
        # 可视化平面内的对准结果
        pc[:, :3] = np.dot(
            rot_2d,
            (pc[:, :3] + trans_2d).T
        ).T
        # 反射率二值化
        intensity = np.copy(pc[:, 3])
        intensity[np.where(intensity < intensity_pivot)] = 0
        intensity[np.where(intensity > intensity_pivot)] = 255

        utils.visualize(input_pc, show=False)
        utils.visualize(np.column_stack([pc[:, :3], intensity.reshape(-1, 1)]), show=False)
        utils.visualize(corners, color=(1, 0, 0), show=False)

    # post process corners
    corners = world_corner_post_process(corners, rot_2d, trans_2d, rot_3d, trans_3d)

    # 如果归一化匹配loss过高 舍弃结果
    final_cost = rst.fun / len(pc) * 1e4
    print('\n\nOptimized loss: {}'.format(final_cost))

    if DEBUG > 0:
        for point in corners:
            utils.visualize_text(point, '{}'.format(int(point[3])))
        utils.visualize(corners, color=(1, 1, 1), show=True)
    return corners, final_cost


def camera_chessboard_corner_detection(img):
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCornersSB(gray, (W, H), None)
    # ret, corners = cv2.findChessboardCorners(gray, (W, H), None)
    # 如果找到足够点对，将其存储起来
    if ret:
        # 消除多余的dimension
        corners = np.squeeze(corners)

        # 确保索引是从左下方的角点开始的
        if corners[0, 1] < corners[-1, 1]:
            # 保证首点在末点下方
            corners = corners[::-1, :]
        elif corners[0, 1] == corners[-1, 1]:
            # 保证首点在末点左侧
            if corners[0, 0] > corners[-1, 0]:
                corners = corners[::-1, :]

        if DEBUG > 1:
            vis = np.copy(img)
            # 将角点在图像上显示
            vis = cv2.drawChessboardCorners(vis, (W, H), corners, ret)
            for corner_id in range(len(corners)):
                cv2.circle(vis, tuple(corners[corner_id]), 5, (0, 0, 255), 10)
                cv2.putText(vis, '{}'.format(corner_id), tuple(corners[corner_id] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            cv2.imshow('findCorners', vis)
            cv2.waitKey(0)
        return corners
    else:
        print('ERROR: Corners not found!!!', file=sys.stderr)
        return None


def solve_extrinsic_matrix(corners_world, corners_image, intrinsic_matrix, distortion):
    assert len(corners_world) == len(corners_image)
    # 外参标定
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        corners_world, corners_image, intrinsic_matrix, distortion)

    # 罗德里格斯变换, 将旋转角转换为旋转矩阵
    rotation_m, _ = cv2.Rodrigues(rvec)

    # 拼接最终的外参
    extrinsic_matrix = np.hstack([rotation_m, tvec])
    extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])
    return extrinsic_matrix, rvec, tvec


def corner_detection_task(pc, img):
    # chessboard localization
    if AUTOMATIC_CHESSBOARD_LOCALIZATION:
        ROI_pc = locate_chessboard(pc)
        ROI_pc, ROI_img = np.array(ROI_pc), None
        pc = crop_ROI_pc(pc, ROI_pc)

    if len(pc) < 100:
        print('WARNING: Points Not Enough !!!', file=sys.stderr)
        return None, 1e10, None

    # 获取3D点云中的棋盘格角点
    corners_world, final_cost = pointcloud_chessboard_corner_detection(pc)

    # 获取2D图像中的棋盘格角点
    corners_image = camera_chessboard_corner_detection(img) if img is not None else None

    return corners_world, final_cost, corners_image


def calibration(keep_list=None, preprocess_hook=None):
    # load intrinsic parameters
    intrinsic_matrix = np.loadtxt(os.path.join(root, 'intrinsic'))
    distortion = np.loadtxt(os.path.join(root, 'distortion'))

    # path to store intermediate results
    PARAMETER_PATH = os.path.join(root, 'parameter')
    if not os.path.exists(PARAMETER_PATH):
        os.mkdir(PARAMETER_PATH)

    if RELABEL:
        ROI_list = {}

    # 读取数据并获取所有点云的bounding box
    pointcloud_image_pair_list = utils.load_data_pair(root, load_rois=not AUTOMATIC_CHESSBOARD_LOCALIZATION)
    # pointcloud_image_pair_list = [pointcloud_image_pair_list[8]]

    pc_list = []
    img_list = []
    for idx, (pc_file, img_file, ROI_file) in pointcloud_image_pair_list:
        # label ROI
        full_pc = np.load(pc_file)
        full_img = cv2.imread(img_file)

        # preprocess data
        if preprocess_hook is not None:
            full_pc = preprocess_hook(full_pc)

        if not AUTOMATIC_CHESSBOARD_LOCALIZATION:
            if RELABEL:
                # 如果没有标注 则手工标注大概位置
                ROI_pc = label_pointcloud_marker_interactive_gui(full_pc)
                # crop ROI that may contains chessboard,
                # only for large image that opencv detects chessboard slowly

                # ROI_img = label_image_marker_interactive_gui(full_img)
                ROI_img = None
                # ROI_list.append((list(ROI_pc), list(ROI_img)))  # save ROI
                ROI_list[str(idx)] = (list(ROI_pc), list(ROI_img) if ROI_img else None)  # save ROI
                open(os.path.join(PARAMETER_PATH, 'ROI.json'), 'w').write(json.dumps(ROI_list))
            else:
                ROI_pc, ROI_img = np.loadtxt(ROI_file), None
                print('ROI Loaded', ROI_pc, ROI_img)

            # crop out ROIs of pc and img
            pc = crop_ROI_pc(full_pc, ROI_pc)
            img = crop_ROI_img(full_img, ROI_img)
        else:
            pc, img = full_pc, full_img

        pc_list.append(pc)
        img_list.append(img)

    # TODO when us opencv imshow gui, the multi processing doesn't work, should fix this
    # process corner detection with multi tasks
    if NUM_WORKERS > 0:
        detection_result = []
        pool = multiprocessing.Pool(NUM_WORKERS)
        for idx, (pc, img) in enumerate(zip(pc_list, img_list)):
            # detect 3D and 2D corners
            detection_result.append(
                pool.apply_async(corner_detection_task, args=(pc, img))
            )
        pool.close()
        pool.join()

    # 遍历点云-图像对 获取3D和2D角点
    corners_world_list, corners_image_list = [], []
    available_list = []
    for idx, (pc, img) in enumerate(zip(pc_list, img_list)):
        print('\n\nCalculating frame: {} / {}'.format(idx, len(pc_list)))
        # get detection results
        if NUM_WORKERS > 0:
            corners_world, final_cost, corners_image = detection_result[idx].get()
        else:
            corners_world, final_cost, corners_image = corner_detection_task(pc, img)

        # 保存列表
        corners_world_list.append(corners_world)
        corners_image_list.append(corners_image)

        if final_cost > OPTIMIZE_TH:
            print('Missing 3D corners, skip this pair', file=sys.stderr)
            continue

        if corners_image is None:
            print('Missing 2D corners, skip this pair', file=sys.stderr)
            continue

        # keep available data index
        available_list.append(idx)

    # save world corners and image corners, for calculating reprojection errors later
    pickle.dump((corners_world_list, corners_image_list), open(os.path.join(PARAMETER_PATH, 'corners.pkl'), 'wb'))

    # keep
    if keep_list is not None:
        available_list = list(set(available_list).intersection(set(keep_list)))

    print(len(available_list), len(corners_world_list), len(corners_image_list))
    # concat all points
    corners_world_list = [corners_world_list[idx] for idx in available_list]
    corners_image_list = [corners_image_list[idx] for idx in available_list]

    # train_list = range(len(corners_world_list))
    # corners_world_list = [corners_world_list[idx] for idx in train_list]
    # corners_image_list = [corners_image_list[idx] for idx in train_list]

    corners_world_list = np.row_stack(corners_world_list)
    corners_image_list = np.row_stack(corners_image_list)

    # 求解相机外参
    extrinsic_matrix, rvec, tvec = solve_extrinsic_matrix(
        corners_world_list[:, :3],
        corners_image_list,
        intrinsic_matrix,
        distortion
    )

    # the L2 norm of extrinsic matrix should be small
    if LA.norm(extrinsic_matrix) > 1e3:
        raise Exception('There are outliers in the points, '
                        'check if there is two frames where chessboards are on the same position')
    # 保存外参
    np.savetxt(os.path.join(PARAMETER_PATH, 'extrinsic'), extrinsic_matrix)

    print('EXT:', extrinsic_matrix)
    print('EXT_NORM:', LA.norm(extrinsic_matrix))
    print('DIST:', distortion)
    print('INTRINSIC:', intrinsic_matrix)

    print('=' * 40)
    print('\n\nExtrinsic parameter saved to {}\n'.format(os.path.join(PARAMETER_PATH, 'extrinsic')))
    print('Done.')
    print('=' * 40)
    print('\n\n')

    # calc re-projection error
    err = utils.calc_reprojection_error(root, extrinsic_matrix, intrinsic_matrix, distortion)
    return err


if __name__ == '__main__':
    calibration(keep_list=None)
