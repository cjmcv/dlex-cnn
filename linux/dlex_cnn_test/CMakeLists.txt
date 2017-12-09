# Cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Options
option(USE_OP_TEST  "Build with operator test"  ON) # Should be the same with option in libdlex_cnn.so

# Add definitions by some options
if(USE_OP_TEST)
    add_definitions(-DUSE_OP_TEST)
endif()

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

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb") 
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 

# Include
SET(INCLUDE_LIST ${PROJECT_SOURCE_DIR}/../../core/include
                 ${PROJECT_SOURCE_DIR}/../../core_test
                 )
include_directories(${INCLUDE_LIST})

message(STATUS "include: ${INCLUDE_LIST}")

# Source
file(GLOB_RECURSE SRC_LIST "${PROJECT_SOURCE_DIR}/../../core_test/*.cpp"
                           "${PROJECT_SOURCE_DIR}/../../core_test/*/*.cpp"
                           )
message(STATUS "src: ${SRC_LIST}")

# Build
add_executable(${PROJECT_NAME} ${SRC_LIST})

# Link libraries
set(PROJECT_SDK_DIR ${PROJECT_SOURCE_DIR}/../)
set(LIBS_LIST ${PROJECT_SDK_DIR}/build/libdlex_cnn.so
              )
target_link_libraries(${PROJECT_NAME} ${LIBS_LIST} -pthread)

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
