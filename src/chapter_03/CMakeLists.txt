set(
    SOURCE_FILES

    # samples
    sample/color_to_grayscale.h
    sample/color_to_grayscale.cu
    sample/blur_kernel.h
    sample/blur_kernel.cu
)

add_library(chapter_03 "${SOURCE_FILES}")

find_package(OpenCV REQUIRED)

target_include_directories(chapter_03 PUBLIC shared)

target_link_libraries(chapter_03 PUBLIC shared ${OpenCV_LIBS})