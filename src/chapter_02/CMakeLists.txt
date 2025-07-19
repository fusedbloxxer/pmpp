set(
    SOURCE_FILES

    # exercises
    exercise/exercises.cpp
    exercise/exercises.h

    # samples
    sample/vec_add_kernel.cu
    sample/vec_add_kernel.h
)

add_library(chapter_02 "${SOURCE_FILES}")