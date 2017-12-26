# Cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Options
option(USE_CUDA  "Build with CUDA support"  ON)
option(USE_OP_TEST  "Build with operator test"  ON)

# Add definitions by some options
if(USE_OP_TEST)
    add_definitions(-DUSE_OP_TEST)
endif()
    
# Define project name
project(dlex_cnn)
set(PROJECT_NAME dlex_cnn)

# Add compiler flag
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

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb") 
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAG_DEBUG "-g -G")
    set(CUDA_NVCC_FLAG_RELEASE "-O3 -Wall")
    add_definitions(-DUSE_CUDA)
endif()

# Include
set(INCLUDE_LIST ${PROJECT_SOURCE_DIR}/../core/include
                 )

include_directories(${INCLUDE_LIST})

message(STATUS "include: ${INCLUDE_LIST}")

# Source
file(GLOB_RECURSE SRC_LIST "${PROJECT_SOURCE_DIR}/../core/src/*.cpp"
                           "${PROJECT_SOURCE_DIR}/../core/src/*/*.cpp")

# Build
if(USE_CUDA)
    file(GLOB_RECURSE CUDA_SRC_LIST "${PROJECT_SOURCE_DIR}/../core/src/*.cu"
                                    "${PROJECT_SOURCE_DIR}/../core/src/*/*.cu")
    cuda_add_library(${PROJECT_NAME} SHARED ${SRC_LIST} ${CUDA_SRC_LIST}) 
else()
    add_library(${PROJECT_NAME} SHARED ${SRC_LIST}) 
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES output_name "utils") 

if(USE_CUDA)
    set(CUDA_LIBS_LIST /usr/local/cuda/lib64/libcublas.so 
                       /usr/local/cuda/lib64/libcurand.so)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBS_LIST})
else()
    target_link_libraries(${PROJECT_NAME})
endif()


# Set output path
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
