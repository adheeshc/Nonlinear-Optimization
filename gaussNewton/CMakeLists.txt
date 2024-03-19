cmake_minimum_required(VERSION 3.0.0)
project(gaussNewton)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})

set(source_files
    gaussNewton.cpp
    GaussNewtonOptimizer.cpp
)

add_executable(gaussNewton ${source_files})

target_link_libraries(gaussNewton ${OpenCV_LIBS})
target_link_libraries(gaussNewton Eigen3::Eigen)
target_link_libraries(gaussNewton Sophus::Sophus)
target_link_libraries(gaussNewton ${Pangolin_LIBRARIES})