# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(dlex_cnn)
set(PROJECT_NAME dlex_cnn)

# add compiler flag
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb") 
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 

ADD_DEFINITIONS(-DCPU_ONLY)

# include
SET(INCLUDE_LIST ${PROJECT_SOURCE_DIR}/../core/include
                 )

include_directories(${INCLUDE_LIST})

message(STATUS "include: ${INCLUDE_LIST}")

# source
file(GLOB_RECURSE COMPARE_SRC_LIST "${PROJECT_SOURCE_DIR}/../core/src/*.cpp"
                                   "${PROJECT_SOURCE_DIR}/../core/src/*/*.cpp")

message(STATUS "src: ${COMPARE_SRC_LIST}")

# build
add_library(${PROJECT_NAME} SHARED ${COMPARE_SRC_LIST}) 

set_target_properties(${PROJECT_NAME} PROPERTIES output_name "utils") 

#SET(LIBS_LIST ${3RDPARTY_DIR}/lib/opencv310/libopencv_core.so
#              )

# Link your application with libraries 
target_link_libraries(${PROJECT_NAME} -lpthread)# ${LIBS_LIST}

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
