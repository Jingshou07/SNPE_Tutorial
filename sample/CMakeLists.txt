cmake_minimum_required(VERSION 3.10)
project(YOLOv8s)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

set(OpenCV_DIR "/opt/thundersoft/opencv-4.5.5/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

include_directories(
    ./
    ../include/
    ../yolov8s/
    /usr/include/zdl
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    test-image
    ./main.cpp
    ../yolov8s/SNPETask.cpp
    ../yolov8s/YOLOv8s.cpp
)

target_link_libraries(
    test-image
    pthread
    dl
    ${OpenCV_LIBS}
    /usr/lib/libSNPE.so
)
