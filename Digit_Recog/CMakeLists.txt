cmake_minimum_required (VERSION 3.0)

PROJECT(Chapter6)

set (CMAKE_CXX_STANDARD 11)

# Requires OpenCV
FIND_PACKAGE( OpenCV 4.6.0 REQUIRED )
MESSAGE("OpenCV version : ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

ADD_EXECUTABLE( traindigits traindigits.cpp )
TARGET_LINK_LIBRARIES( traindigits ${OpenCV_LIBS} )

