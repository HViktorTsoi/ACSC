//
// Created by hviktortsoi on 20-5-27.
//
#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <pcl/segmentation/impl/region_growing.hpp>
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZI PointCloudType;
typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudPtr;
typedef std::vector<std::vector<int>> ClusterIndices;

struct SegmentationParameter {
    bool ground_removal;

    unsigned int number_of_neighbours;
    int max_region_size;
    int min_region_size;
    // (in deg)
    float smoothness_threshold;
    float curvature_threshold;

    SegmentationParameter(bool groundRemoval, unsigned int numberOfNeighbours, int minRegionSize,
                          int maxRegionSize, float smoothnessThreshold, float curvatureThreshold)
            : ground_removal(groundRemoval),
              number_of_neighbours(numberOfNeighbours),
              min_region_size(minRegionSize),
              max_region_size(maxRegionSize),
              smoothness_threshold(smoothnessThreshold),
              curvature_threshold(curvatureThreshold) {}

};


std::vector<pcl::PointIndices> region_growing(
        const PointCloudPtr &cloud,
        SegmentationParameter parameter,
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored_cloud = nullptr
) {

    pcl::search::Search<PointCloudType>::Ptr tree(new pcl::search::KdTree<PointCloudType>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // filter groud
    if (parameter.ground_removal) {
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<PointCloudType> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.2);
        seg.setMaxIterations(200);

        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        pcl::ExtractIndices<PointCloudType> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud);

    }

    // remove outliers
    pcl::StatisticalOutlierRemoval<PointCloudType> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(0.3);
    sor.filter(*cloud);

    // calc normals
    pcl::NormalEstimation<PointCloudType, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(50);
    ne.compute(*normals);

    // segmentation
    pcl::RegionGrowing<PointCloudType, pcl::Normal> reg;
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(parameter.number_of_neighbours);

    reg.setMinClusterSize(parameter.min_region_size);
    reg.setMaxClusterSize(parameter.max_region_size);
    reg.setSmoothnessThreshold(parameter.smoothness_threshold / 180.0 * M_PI);
    reg.setCurvatureThreshold(parameter.curvature_threshold);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    if (colored_cloud) {
        *colored_cloud = *reg.getColoredCloud();
    }

    return clusters;
}
