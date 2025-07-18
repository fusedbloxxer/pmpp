cmake_minimum_required(VERSION 3.26)

project(pmpp VERSION 0.0.0 DESCRIPTION "Learning CUDA with C++" LANGUAGES CXX CUDA)

# Set build type
set(CMAKE_BUILD_TYPE Debug)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C++ standard for CUDA
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Create output target
add_executable(pmpp src/main.cu)

# Set CUDA architecture to native for the current machine
set_property(TARGET pmpp PROPERTY CUDA_ARCHITECTURES native)

# Enable verbose warnings for both host and device code and enable debug information
target_compile_options(pmpp PRIVATE -Wall -Wextra)