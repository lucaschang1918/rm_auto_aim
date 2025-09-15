cmake_minimum_required(VERSION 3.16)
project(rm_auto_aim)

set(CMAKE_CXX_STANDARD 20)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 OpenVINO
find_package(OpenVINO REQUIRED)

# 头文件目录
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenVINO_INCLUDE_DIRS})


set(SOURCES
        src/detector_node.cpp
        src/detector.cpp
        src/number_classifier.cpp
        src/pnp_solver.cpp
)

# 可执行文件
add_executable(rm_auto_aim ${SOURCES})

# 链接 OpenCV + OpenVINO
target_link_libraries(rm_auto_aim PRIVATE ${OpenCV_LIBS} openvino::runtime)
