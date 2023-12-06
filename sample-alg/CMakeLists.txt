cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
PROJECT(AlgYolov8s-3rd)

include(FindPkgConfig)
set(OpenCV_DIR "/opt/thundersoft/opencv-4.5.5/lib/cmake/opencv4")
find_package(OpenCV         REQUIRED)
pkg_check_modules(GST       REQUIRED gstreamer-1.0)
pkg_check_modules(JSON      REQUIRED json-glib-1.0)
pkg_check_modules(GLIB      REQUIRED glib-2.0)
pkg_check_modules(LOG4CPLUS REQUIRED log4cplus)
pkg_check_modules(GFLAGS    REQUIRED gflags)

message(STATUS "GST:      ${GST_INCLUDE_DIRS},${GST_LIBRARIES}")
message(STATUS "JSON:     ${JSON_INCLUDE_DIRS},${JSON_LIBRARIES}")
message(STATUS "GLIB:     ${GLIB_INCLUDE_DIRS},${GLIB_LIBRARIES}")
message(STATUS "OpenCV:   ${OpenCV_INCLUDE_DIRS},${OpenCV_LIBRARIES}")
message(STATUS "LOG4CPLUS:${LOG4CPLUS_INCLUDE_DIRS},${LOG4CPLUS_LIBRARIES}")
message(STATUS "GFLAGS:   ${GFLAGS_INCLUDE_DIRS},${GFLAGS_LIBRARIES}")

include_directories(
    ./
    ../include/
    ../yolov8s/
    /usr/include/zdl
    ${GST_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${LOG4CPLUS_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
)

link_directories(
    /opt/thundersoft/algs/lib
)

add_library(
    AlgYolov8s-3rd SHARED
    AlgYolov8s-3rd.cpp
    ../yolov8s/YOLOv8s.cpp
    ../yolov8s/SNPETask.cpp
)

target_link_libraries(
    AlgYolov8s-3rd
    ${GST_LIBRARIES}
    ${JSON_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${OpenCV_LIBRARIES}
    pthread
    dl
    /usr/lib/libSNPE.so
)

install(
    TARGETS AlgYolov8s-3rd
    LIBRARY DESTINATION /opt/thundersoft/algs/lib
)

install(
    FILES AlgYolov8s-3rd.json
    DESTINATION /opt/thundersoft/configs
)

install(
    FILES ../models/yolov8s_200-epochs_640x640_snpe1.61_quantize.dlc
    DESTINATION /opt/thundersoft/algs/model
)

