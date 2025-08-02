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

# List all source files
set(
    SOURCE_FILES

    # main
    src/main.cpp

    # shared
    src/shared/shared.cpp
    src/shared/shared.h

    # chapter_02
    src/chapter_02/exercise/exercise.cpp
    src/chapter_02/exercise/exercise.h
    src/chapter_02/sample/vec_add_kernel.cu
    src/chapter_02/sample/vec_add_kernel.h
    src/chapter_02/chapter_02.h

    # chapter_03
    src/chapter_03/sample/blur_kernel.cu
    src/chapter_03/sample/blur_kernel.h
    src/chapter_03/sample/color_to_grayscale.cu
    src/chapter_03/sample/color_to_grayscale.h
    src/chapter_03/exercise/exercise.h
    src/chapter_03/exercise/exercise.cpp
    src/chapter_03/exercise/mm_kernel.cu
    src/chapter_03/exercise/mm_kernel.h
    src/chapter_03/exercise/mm.cpp
    src/chapter_03/exercise/mm.h
    src/chapter_03/chapter_03.h

    # chapter_04
    src/chapter_04/sample/cuda_device_props.cu
    src/chapter_04/sample/cuda_device_props.h
    src/chapter_04/exercise/exercise.cpp
    src/chapter_04/exercise/exercise.h
    src/chapter_04/chapter_04.h
)

# Create executable target from source files
add_executable(pmpp ${SOURCE_FILES})

# Set CUDA to build for the native host GPU
set_property(TARGET pmpp PROPERTY CUDA_ARCHITECTURES native)

# Set where OpenCV_ROOT can be found
set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")

# Add external libraries
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Add headers such that the libs can be referenced in the code
target_include_directories(pmpp PUBLIC "${OpenCV_INCLUDE_DIRS}")

# Compile options for the target executable
target_compile_options(pmpp PUBLIC -Wall -Wextra -O0)

# Link the libraries such that they can be referenced in the code
target_link_libraries(pmpp PUBLIC ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# Copy resources to buiild output directory
file(COPY src/resources DESTINATION .)