cmake_minimum_required(VERSION 3.5)
project(RegionGrowingSegmentation)
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(segmentation_ext segmentation.cpp segmentation.hpp)
set_target_properties(segmentation_ext PROPERTIES PREFIX "")
set_target_properties(segmentation_ext PROPERTIES SUFFIX ".so")
target_link_libraries(segmentation_ext PRIVATE ${PCL_LIBRARIES})
