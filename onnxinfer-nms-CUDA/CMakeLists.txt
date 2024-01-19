cmake_minimum_required(VERSION 3.10)

project(yolov5)

add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
set(ONNXRUNTIME_DIR "/home/a/env/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
include_directories(${PROJECT_SOURCE_DIR}/include/)
# include_directories(${PROJECT_SOURCE_DIR}/plugins/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugins/*.cu  ${PROJECT_SOURCE_DIR}/plugins/*.h)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5 ${SRCS})
target_link_libraries(yolov5  nvinfer)
target_link_libraries(yolov5  cudart)
target_link_libraries(yolov5  myplugins)
target_link_libraries(yolov5  ${OpenCV_LIBS})
target_link_libraries(yolov5 "/home/a/env/onnxruntime/lib/libonnxruntime.so")




