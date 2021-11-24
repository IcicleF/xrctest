cmake_minimum_required(VERSION 3.10)
project(Plachta)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wno-sign-compare -O2 -march=native -fopenmp")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Configure MPI
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_executable(main main.cpp)
target_link_libraries(main ibverbs pthread ${MPI_CXX_LIBRARIES})