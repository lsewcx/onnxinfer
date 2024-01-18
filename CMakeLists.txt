cmake_minimum_required(VERSION 3.1)

# 项目名称
project(onnxinfer)
set (CMAKE_CXX_STANDARD 17)


# 找到 OpenCV 库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(ONNXRUNTIME_DIR "/home/a/env/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
# 添加源代码文件
add_executable(onnxinfer main.cpp YOLOV5.cpp)

# 链接 ONNX Runtime、OpenCV 和 CUDA 库
target_link_libraries(onnxinfer ${OpenCV_LIBS} "/home/a/env/onnxruntime/lib/libonnxruntime.so")
