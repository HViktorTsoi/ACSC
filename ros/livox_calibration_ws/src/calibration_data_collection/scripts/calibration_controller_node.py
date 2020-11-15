#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import, print_function

import json
import os
import shutil
import sys
# import thread
import thread

import cv2
import numpy as np
import ros_numpy
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, Image
import time
import yaml
import argparse

print(sys.argv)
# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, default='', help='path to config path', required=True)
parser.add_argument('--lidar-id', type=int, default=-1, help='lidar id, for multi lidar calibration')
parser.add_argument('--data-saving-path', type=str, required=True, help='path to save calibration image and pointcloud')
parser.add_argument('--image-topic', type=str, required=True, help='topic name for receive image')
parser.add_argument('--lidar-topic', type=str, required=True, help='topic name for receive lidar point cloud')
parser.add_argument('--data-tag', type=str, help='data tag')
parser.add_argument('--overlap', type=int, required=True, help='overlapping frames for lidar')
parser.add_argument('args', nargs=argparse.REMAINDER)
parser.parse_args()

args = parser.parse_args()

try:
    # configs = yaml.load(open(args.config_file, 'r').read(), Loader=yaml.FullLoader)
    configs = yaml.load(open(args.config_file, 'r').read())
except Exception as e:
    print(e)
    rospy.logerr('Wrong config path: "{}"'.format(sys.argv[1]))
    raise IOError('Wrong config path: "{}"'.format(sys.argv[1]))
# scene size when labeling chessboards
BOUND = configs['BOUND']  # xmin,xmax,ymin,ymax,zmin,zmax

# whether use automatic calibration
AUTOMATIC_CHESSBOARD_LOCALIZATION = configs['AUTOMATIC_CHESSBOARD_LOCALIZATION']

# resolution of displayed point cloud projection, adjust it according to your screen size
VIRTUAL_CAM_INTRINSIC = configs['VIRTUAL_CAM_INTRINSIC']  # 虚拟相机内参

# how many lines of log are displayed on chessboard labeler GUI
LOG_DISPLAY_LINES = configs['LOG_DISPLAY_LINES']

# bird view shape
IMG_SHAPE = (int((BOUND[1] - BOUND[0]) * VIRTUAL_CAM_INTRINSIC),
             int((BOUND[3] - BOUND[2]) * VIRTUAL_CAM_INTRINSIC))  # height, width

IMG_TOPIC = args.image_topic
POINTCLOUD_TOPIC = args.lidar_topic

# path to save calibration data
DATA_TAG = args.data_tag.strip() if args.data_tag else time.time()
ROOT_PATH = '{}/{}{}'.format(args.data_saving_path, DATA_TAG,
                             '_{:04d}'.format(args.lidar_id) if args.lidar_id != -1 else '')
PCD_PATH = os.path.join(ROOT_PATH, 'pcds')
IMG_PATH = os.path.join(ROOT_PATH, 'images')
ROI_PATH = os.path.join(ROOT_PATH, 'ROIs')

# overlapped frames for pointclouds
# PCD_OVERLAPPING_FRAMES = configs['PCD_OVERLAPPING_FRAMES']
PCD_OVERLAPPING_FRAMES = args.overlap

# ================== global variable
global_pc = np.empty([1, 5])
global_img = None

global_frame_id = 0
global_start = False
overlaped_pcd = []

if AUTOMATIC_CHESSBOARD_LOCALIZATION:
    global_gui_text = [
        '<Wheel Up/Down> to zoom in/out.',
        'Press <Enter> to record.',
    ]
else:
    global_gui_text = [
        '<Wheel Up/Down> to zoom in/out.',
        '<Double Click> to draw box framing the chessboard.',
    ]


# ============================================Render
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

    def __init__(self, name, img_shape):
        self._drawing = False
        self._start = (-1, -1)
        self._end = (-1, -1)
        self._marker_layer = np.zeros(list(img_shape) + [3], dtype=np.uint8)
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
            cv2.rectangle(self._marker_layer, self._start, self._end, (255, 255, 255), 1)
            # 获取bb结果
            self.BB = (self._start[0], self._start[1], self._end[0], self._end[1])
            log_global_info('BB: {}'.format(
                np.round(convert_BB_2d3d(np.array(self.BB)), decimals=2)
            ))
            log_global_info('Press <Enter> to record.')

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

    def render(self, bg_img, text_list=[]):
        """
        渲染
        :return:
        """
        # combine layer
        img_vis = cv2.bitwise_or(bg_img, self._marker_layer)

        # draw multi-line text
        for idx, txt in enumerate(text_list):
            cv2.putText(img_vis, txt, (10, 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1)
        cv2.imshow(self._name, img_vis)


def log_global_info(text, level=sys.stdout):
    global global_gui_text
    global_gui_text.append(text)
    # only display the latest 5 frames
    global_gui_text = global_gui_text[-LOG_DISPLAY_LINES:]

    print(text, file=level)


def thread_render_bird_view():
    global global_pc, global_start, global_gui_text
    # 交互绘图
    labeler_bird = BoxLabeler('LiDAR ID:{} Chessboard Labeler on Birdeye-view'.format(args.lidar_id), IMG_SHAPE)

    # 执行渲染
    while not rospy.is_shutdown():
        # 转换为鸟瞰图投影
        if global_pc is not None:
            bird_view = bird_view_projection(global_pc)
        else:
            bird_view = np.zeros(IMG_SHAPE + (3,), dtype=np.uint8)

        # 渲染
        labeler_bird.render(bird_view, text_list=global_gui_text)
        if global_img is not None:
            cv2.imshow('image', cv2.resize(global_img, dsize=(600, 320)))

        # 获取标定结果
        if cv2.waitKey(1) == 13:
            # if start labeling, disable the enter call back
            if global_start:
                continue
            if not AUTOMATIC_CHESSBOARD_LOCALIZATION:
                # 保证获取了所有标定
                if labeler_bird.BB is None:
                    log_global_info('*** There is NO valid bounding box!!! ***', level=sys.stderr)
                    continue
                # save BB
                BB = convert_BB_2d3d(np.array(labeler_bird.BB))
                np.savetxt(os.path.join(ROI_PATH, '{:06d}.txt'.format(global_frame_id)), BB)
                log_global_info('BB saved. {}'.format(BB))

            # start to collect pc and img
            global_start = True
            log_global_info('Start to collect data...')

            # reset bounding box
            labeler_bird.BB = None
    else:
        print('Rendering thread is shutting down...')


def convert_BB_2d3d(BB_bird):
    # 还原到world坐标系
    BB_bird = BB_bird / float(VIRTUAL_CAM_INTRINSIC)
    BB_bird[[0, 2]] += BOUND[2]  # 鸟瞰图宽度方向，对应点云y轴
    BB_bird[[1, 3]] += BOUND[0]  # 鸟瞰图高度方向，对应点云x轴

    # 3D Bounding Box
    X_MIN = min(BB_bird[[1, 3]])
    X_MAX = max(BB_bird[[1, 3]])
    Y_MIN = min(BB_bird[[0, 2]])
    Y_MAX = max(BB_bird[[0, 2]])
    Z_MIN = BOUND[-2]
    Z_MAX = BOUND[-1]

    return X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX


def bird_view_projection(pc):
    # 限制范围
    pc = pc[np.where(
        (BOUND[0] < pc[:, 0]) & (pc[:, 0] < BOUND[1]) &
        (BOUND[2] < pc[:, 1]) & (pc[:, 1] < BOUND[3]) &
        (BOUND[4] < pc[:, 2]) & (pc[:, 2] < BOUND[5])
    )]

    img_bird = np.zeros(IMG_SHAPE + (3,), dtype=np.uint8)

    # 投影
    bird_x = np.int_((pc[:, 1] - BOUND[2]) * VIRTUAL_CAM_INTRINSIC)
    bird_y = np.int_((pc[:, 0] - BOUND[0]) * VIRTUAL_CAM_INTRINSIC)

    img_bird[bird_y, bird_x, 2] = pc[:, 2] * VIRTUAL_CAM_INTRINSIC
    img_bird[bird_y, bird_x, 1] = pc[:, 3] * 10

    return img_bird


# ============================================ROS
def cb_image(img):
    global global_img
    # store img to global scope
    img = bridge.imgmsg_to_cv2(img, "passthrough")
    global_img = img


def cb_pointcloud(input_pointcloud):
    global global_start, global_frame_id, \
        global_pc, global_img, \
        overlaped_pcd

    pc_array = ros_numpy.numpify(input_pointcloud)
    if len(pc_array.shape) == 1:
        pc = np.zeros((pc_array.shape[0], 5))
        # 解析点格式
        pc[:, 0] = pc_array['x']
        pc[:, 1] = pc_array['y']
        pc[:, 2] = pc_array['z']
        pc[:, 3] = pc_array['intensity']
    elif len(pc_array.shape) == 2:
        pc = np.zeros((pc_array.shape[0] * pc_array.shape[1], 5))
        # 解析点格式
        pc[:, 0] = pc_array['x'].reshape(-1)
        pc[:, 1] = pc_array['y'].reshape(-1)
        pc[:, 2] = pc_array['z'].reshape(-1)
        pc[:, 3] = pc_array['intensity'].reshape(-1)
    else:
        raise Exception('Unsupported Pointcloud2 Format!!!')
    pc[:, 4] = len(overlaped_pcd)

    # 传递给全局信息以供render 同时用滑动窗口叠加更多的点
    global_pc = np.concatenate([global_pc, pc])[-len(pc) * 5:]

    # 处理点云叠加
    if global_start:
        overlaped_pcd.append(pc)
        log_global_info('Recording pcd {}... {} {}'.format(global_frame_id, len(overlaped_pcd), pc.shape))

        # finish recording
        if len(overlaped_pcd) == PCD_OVERLAPPING_FRAMES:
            # concat pcd
            overlaped_pcd = np.row_stack(overlaped_pcd)

            # save pcd and img
            np.save(os.path.join(PCD_PATH, '{:06d}.npy'.format(global_frame_id)), overlaped_pcd)
            log_global_info('PCD saved.')

            cv2.imwrite(os.path.join(IMG_PATH, '{:06d}.png'.format(global_frame_id)), global_img)
            log_global_info('IMG saved.')

            # reset flags
            global_start = False
            overlaped_pcd = []
            global_frame_id += 1

            log_global_info('Recording done.')
            log_global_info('<Wheel Up/Down> to zoom in/out.')
            if not AUTOMATIC_CHESSBOARD_LOCALIZATION:
                log_global_info('<Double Click> to draw box framing the chessboard.', )
            else:
                log_global_info('Press <Enter> to record.', )


if __name__ == '__main__':
    # 渲染线程
    thread.start_new_thread(thread_render_bird_view, ())

    # Ros相关
    rospy.init_node('Calibration_data_collection_controller{}'.format('_LiDAR{:04d}'.format(args.lidar_id) if args.lidar_id != -1 else ''))
    print('Init......')

    # prepare file path
    if not os.path.exists(os.path.join(ROOT_PATH)):
        os.mkdir(ROOT_PATH)

    os.mkdir(PCD_PATH)
    os.mkdir(IMG_PATH)
    os.mkdir(ROI_PATH)

    bridge = CvBridge()

    img_subscriber = rospy.Subscriber(IMG_TOPIC, Image, cb_image)
    lidar_subscriber = rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, cb_pointcloud)

    rospy.spin()
