cmake_minimum_required (VERSION 3.0)

PROJECT(Chapter6)

set (CMAKE_CXX_STANDARD 11)

# Requires OpenCV
find_package(OpenCV 4.6.0 REQUIRED )
message("OpenCV version : ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_executable( erosion erosion.cpp )
target_link_libraries( erosion ${OpenCV_LIBS} )
