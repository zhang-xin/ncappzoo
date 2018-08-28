project(simple_classifier)
cmake_minimum_required(VERSION 2.8)

include_directories(
	$ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/include
	$ENV{INTEL_CVSDK_DIR}/opencv/include/opencv2
	)

set(SOURCES
    run.cpp
)

add_executable(simple_classifier ${SOURCES})

link_directories($ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64 /usr/lib/x86_64-linux-gnu/)
find_library(INFERENCE_ENGINE_LIB inference_engine $ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64)

target_link_libraries(simple_classifier ${INFERENCE_ENGINE_LIB} opencv_core opencv_highgui opencv_imgproc dl)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
