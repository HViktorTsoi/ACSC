from __future__ import print_function, division, absolute_import
import pickle
import time

from scipy import optimize
import transforms3d
import numpy as np
import numpy.linalg as LA
from mayavi import mlab
import utils
import matplotlib.pyplot as plt
import pcl


def roate_with_rt(r_t, points):
    rot_mat = transforms3d.quaternions.quat2mat(r_t[:4])
    # rot_mat = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], r_t[2]),
    #                  np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], r_t[1]),
    #                         transforms3d.axangles.axangle2mat([1, 0, 0], r_t[0])))

    transformed_pcd = np.dot(rot_mat, points.T).T + r_t[[4, 5, 6]]
    return transformed_pcd


def cost_func(rot_trans_mat, points_from, points_to):
    points_to_estimate = roate_with_rt(rot_trans_mat, points_from)
    cost = LA.norm(points_to.reshape(-1) - points_to_estimate.reshape(-1), ord=2)
    # print('cost: ', cost)
    return cost


def icp_extrinsic():
    frame_id = 1
    livox = np.load('data/20200708_wto_lidar_calibdata/1594219638.9_0000/pcds/{:06d}.npy'.format(frame_id))
    rs128 = np.load('data/20200708_wto_lidar_calibdata/1594219638.84_0001/pcds/{:06d}.npy'.format(frame_id))
    rs128 = rs128[np.where(rs128[:, 0] > 0)]

    livox = utils.voxelize(livox, voxel_size=0.05)
    rs128 = utils.voxelize(rs128, voxel_size=0.05)

    pc_from = pcl.PointCloud(livox[:, :3].astype(np.float32))
    pc_to = pcl.PointCloud(rs128[:, :3].astype(np.float32))

    icp = pc_from.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(pc_from, pc_to, max_iter=1000)

    print(converged, transf, estimate, fitness)

    estimate = estimate.to_array()

    mlab.points3d(estimate[:, 0], estimate[:, 1], estimate[:, 2], color=(1, 0, 0), mode='point')
    mlab.points3d(rs128[:, 0], rs128[:, 1], rs128[:, 2], color=(0, 1, 0), mode='point')

    mlab.axes()
    mlab.show()


if __name__ == '__main__':

    # icp_extrinsic()
    # exit()

    # world_corners_livox, _ = pickle.load(open('./data/20200708_wto_lidar_calibdata/1594219638.9_0000/parameter/corners.pkl', 'rb'))
    # world_corners_128, _ = pickle.load(open('./data/20200708_wto_lidar_calibdata/1594219638.84_0001/parameter/corners.pkl', 'rb'))
    world_corners_from, _ = pickle.load(open('./data/20200709_two_livox/1594299937.47_0000/parameter/corners.pkl', 'rb'))
    world_corners_to, _ = pickle.load(open('./data/20200709_two_livox/1594299937.44_0001/parameter/corners.pkl', 'rb'))
    # world_corners_from, _ = pickle.load(open('./data/20200714_BIGBOARD/1594733004.9_0000/parameter/corners.pkl', 'rb'))
    # world_corners_to, _ = pickle.load(open('./data/20200714_BIGBOARD/1594733004.9_0001/parameter/corners.pkl', 'rb'))

    points_from = np.row_stack(world_corners_from)
    points_to = np.row_stack(world_corners_to)

    tic = time.time()
    keep_indices = None
    for keep_th in [0.1, 0.05, 0.04, 0.03]:
        if keep_indices is not None:
            points_from = points_from[keep_indices]
            points_to = points_to[keep_indices]
        assert len(points_from) == len(points_to)
        print('\n\nPoints remain: ', len(points_from))

        # optimize rigid transform
        initial_guess = np.array([0, 0, 0, 1, 0, 0, 0])
        rst = optimize.minimize(cost_func, initial_guess, args=(points_from[:, :3], points_to[:, :3]))
        rot_trans_mat = rst.x
        toc = time.time()
        print('Time used: ', toc - tic)
        print('Loss: ', rst.fun)

        # projection to valid
        points_to_estimate = np.copy(points_from)
        points_to_estimate[:, :3] = roate_with_rt(rot_trans_mat, points_from[:, :3])
        residual = LA.norm(points_to - points_to_estimate, ord=2, axis=1)
        # plt.hist(residual[np.where(residual > 0.1)], bins=50)
        plt.hist(residual, bins=50)
        plt.show()

        # keep inliers
        keep_indices = np.where(residual < keep_th)

    print('Final 6D transform')
    print('RPY: ', transforms3d.quaternions.quat2axangle(rot_trans_mat[:4]))
    print('XYZ: ',rot_trans_mat[4:])
    # # vis
    # utils.visualize(points_from, color=(0, 0, 1))
    # utils.visualize(points_to, color=(1, 0, 0))
    # for point in points_to:
    #     utils.visualize_text(point, '{}'.format(int(point[3])))
    # utils.visualize(points_to_estimate, color=(0, 1, 0))
    # for point in points_to_estimate:
    #     utils.visualize_text(point, '{}'.format(int(point[3])))
    # mlab.show()

    for frame_id in range(28):
        # lidar_from = np.load('./data/20200714_BIGBOARD/1594733004.9_0000/pcds/{:06d}.npy'.format(frame_id))
        # lidar_to = np.load('./data/20200714_BIGBOARD/1594733004.9_0001/pcds/{:06d}.npy'.format(frame_id))
        lidar_from = np.load('./data/20200709_two_livox/1594299937.47_0000//pcds/{:06d}.npy'.format(frame_id))
        lidar_to = np.load('./data/20200709_two_livox/1594299937.44_0001//pcds/{:06d}.npy'.format(frame_id))

        lidar_from = lidar_from[np.where((lidar_from[:, 0] > 0) & (lidar_from[:, 0] < 50))]
        lidar_to = lidar_to[np.where((lidar_to[:, 0] > 0) & (lidar_to[:, 0] < 50))]

        livox_trans = np.copy(lidar_from)
        livox_trans[:, :3] = roate_with_rt(rot_trans_mat, lidar_from[:, :3])

        mlab.points3d(livox_trans[:, 0], livox_trans[:, 1], livox_trans[:, 2], color=(1, 0, 0), mode='point')
        mlab.points3d(lidar_to[:, 0], lidar_to[:, 1], lidar_to[:, 2], color=(0, 1, 0), mode='point')
        # mlab.points3d(livox_trans[:, 0], livox_trans[:, 1], livox_trans[:, 2], livox_trans[:, 3] * 2, mode='point')
        # mlab.points3d(rs128[:, 0], rs128[:, 1], rs128[:, 2], rs128[:, 3], mode='point')
        mlab.show()

        # fusion = np.row_stack([livox_trans, lidar_to])
        # fusion = utils.voxelize(fusion, voxel_size=0.04)
        #
        # print(fusion.shape)
        # np.save('./data/20200714_BIGBOARD/fusion/pcds/{:06d}.npy'.format(frame_id), fusion)
        # mlab.points3d(fusion[:, 0], fusion[:, 1], fusion[:, 2], fusion[:, 3], mode='point')
        # mlab.show()
