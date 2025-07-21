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
set_property(TARGET pmpp PROPERTY CUDA_ARCHITECTURES native)

# Add subprojects
add_subdirectory(src/chapter_02)
add_subdirectory(src/chapter_03)

# Link libraries to the target
target_link_libraries(pmpp PUBLIC chapter_02)
target_link_libraries(pmpp PUBLIC chapter_03)
target_include_directories(pmpp PUBLIC "src/chapter_02" PUBLIC "src/chapter_03")

# Set compile options for the project
target_compile_options(pmpp PRIVATE -Wall -Wextra)