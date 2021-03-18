cmake_minimum_required(VERSION 3.17)
project(PAvx2_Sort)

set(CMAKE_CXX_STANDARD 17)

set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++17 -Ofast -march=native -mavx2 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(PAvx2_Sort main.cpp)