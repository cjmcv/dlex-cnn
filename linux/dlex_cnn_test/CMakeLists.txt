# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(dlex_cnn_test)
set(PROJECT_NAME dlex_cnn_test)

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

#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb") 
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 

ADD_DEFINITIONS(-DCPU_ONLY)

# include
SET(INCLUDE_LIST ${PROJECT_SOURCE_DIR}/../../core/include
                 ${PROJECT_SOURCE_DIR}/../../core_test
                 )
include_directories(${INCLUDE_LIST})

message(STATUS "include: ${INCLUDE_LIST}")

set(PROJECT_SDK_DIR ${PROJECT_SOURCE_DIR}/../)
SET(LIBS_LIST ${PROJECT_SDK_DIR}/build/libdlex_cnn.so
              )

# source
file(GLOB_RECURSE COMPARE_SRC_LIST "${PROJECT_SOURCE_DIR}/../../core/src/*.cpp"
                                   "${PROJECT_SOURCE_DIR}/../../core/src/*/*.cpp"
                                   "${PROJECT_SOURCE_DIR}/../../core_test/*.cpp"
                                   "${PROJECT_SOURCE_DIR}/../../core_test/*/*.cpp"
                                   )

message(STATUS "src: ${COMPARE_SRC_LIST}")

# build
add_executable(${PROJECT_NAME} ${COMPARE_SRC_LIST})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${LIBS_LIST} -lpthread) # -lpthread -ltcmalloc

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
