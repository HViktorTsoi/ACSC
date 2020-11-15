//
// Created by hviktortsoi on 20-5-27.
//
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <algorithm>
#include "segmentation.hpp"

namespace py=pybind11;


py::array_t<float> region_growing_kernel(
        const py::array_t<float> &input,
        bool groundRemoval,
        unsigned int numberOfNeighbours,
        int minRegionSize,
        int maxRegionSize,
        float smoothnessThreshold,
        float curvatureThreshold
) {
    auto ref_input = input.unchecked<2>();
    // 初始化pointcloud 数量是输入的numpy array中的point数量
    PointCloudPtr cloud(new PointCloud(ref_input.shape(0), 1));
//#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < ref_input.shape(0); ++i) {
        cloud->points[i].x = ref_input(i, 0);
        cloud->points[i].y = ref_input(i, 1);
        cloud->points[i].z = ref_input(i, 2);
        cloud->points[i].intensity = ref_input(i, 3);
    }
//    std::cout << "INPUT SHAPE: " << ref_input.shape(0) << " " << ref_input.shape(1) << std::endl;

    // segmentation
//    SegmentationParameter parameter(true, 100, 500, 1500, 5.0, 0.2);
    SegmentationParameter parameter(
            groundRemoval, numberOfNeighbours, minRegionSize,
            maxRegionSize, smoothnessThreshold, curvatureThreshold
    );
    auto clusters = region_growing(cloud, parameter);

    // calc available point count
    int useful_point_count = 0;
    for (auto &indices:clusters) {
        useful_point_count += indices.indices.size();
    }

    // results
    int data_field = 5;
    auto result = py::array_t<float>(py::array::ShapeContainer(
            {(const long) useful_point_count, data_field}
    ));
//    std::cout << "RESULT SHAPE: " << result.shape(0) << " " << result.shape(1) << std::endl;

    float *buf = (float *) result.request().ptr;
    // output point id
    int point_id = 0;
    for (int label_id = 0; label_id < clusters.size(); ++label_id) {
        for (auto &indice:clusters[label_id].indices) {
            int buf_index_base = point_id * data_field;
            buf[buf_index_base + 0] = cloud->points[indice].x;
            buf[buf_index_base + 1] = cloud->points[indice].y;
            buf[buf_index_base + 2] = cloud->points[indice].z;
            buf[buf_index_base + 3] = cloud->points[indice].intensity;
            buf[buf_index_base + 4] = label_id;
            point_id++;
        }
    }
    return result;
}

PYBIND11_MODULE(segmentation_ext, m) {
    m.doc() = "region growing segmentation";

    m.def("region_growing_kernel", &region_growing_kernel, "region growing segmentation");

}
