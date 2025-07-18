cmake_minimum_required(VERSION 3.18)

# Define project
project(pmpp VERSION 0.0.0 DESCRIPTION "Learning CUDA with C++" LANGUAGES C CXX CUDA)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Create output target
add_executable(pmpp src/main.cu)
