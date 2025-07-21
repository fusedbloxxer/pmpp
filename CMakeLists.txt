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
add_executable(pmpp src/main.cpp)

# Set CUDA to build for the native host GPU
set_property(TARGET pmpp PROPERTY CUDA_ARCHITECTURES native)

# Set where OpenCV_ROOT can be found
set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")

# Find each chapter's cmake file
add_subdirectory(src/chapter_02)
add_subdirectory(src/chapter_03)
add_subdirectory(src/shared)

# Find a package using config mode (<PackageName>Config.cmake)
find_package(OpenCV REQUIRED)

# Add headers such that the libs can be referenced in the code
target_include_directories(pmpp PUBLIC "src/shared")
target_include_directories(pmpp PUBLIC "src/chapter_03")
target_include_directories(pmpp PUBLIC "src/chapter_02")
target_include_directories(pmpp PUBLIC "${OpenCV_INCLUDE_DIRS}")

# Compile options for the target executable
target_compile_options(pmpp PUBLIC -Wall -Wextra)

# Link the libraries such that they can be referenced in the code
target_link_libraries(pmpp PUBLIC shared chapter_02 chapter_03 ${OpenCV_LIBS})

# Copy resources to buiild output directory
file(COPY src/resources DESTINATION .)