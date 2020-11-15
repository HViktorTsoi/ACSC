//
// Created by hvt on 19-8-13.
//
#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/icp.h>

ros::Publisher publisher;

void cb_overlap(const sensor_msgs::PointCloud2 &cloud_msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(cloud_msg, *cloud);

    // denoise for each incoming frame to be integrated
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(30);
    sor.setStddevMulThresh(2);
//    sor.setNegative(true);
    sor.filter(*cloud);

    // 返回结果
    pcl::PCLPointCloud2 output;
    pcl::toPCLPointCloud2(*cloud, output);

    // 从最近一帧复制header
    pcl::PCLHeader latest_header = pcl_conversions::toPCL(cloud_msg.header);
    output.header.stamp = latest_header.stamp;
    output.header.frame_id = latest_header.frame_id;
    output.header.seq = latest_header.seq;

    publisher.publish(output);


}

int main(int argc, char **argv) {
    ros::init(argc, argv, "livox_integration");
    ros::NodeHandle nh;

//    ros::Subscriber subscriber = nh.subscribe("/livox/lidar", 10, cb_overlap);
    ROS_INFO("================================");
    ROS_INFO(argv[1]);
    ros::Subscriber subscriber = nh.subscribe(argv[1], 10, cb_overlap);
    publisher = nh.advertise<sensor_msgs::PointCloud2>("/livox/to_integrate", 1);

    ros::spin();
}