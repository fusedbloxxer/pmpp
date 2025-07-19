#include "exercises.h"

#include <iostream>

int chapter_02::exercise::main()
{
    auto exercise_01 =
        "1. If we want to use each thread in a grid to calculate one output element of a vector addition,"
        "what would be the expression for mapping the thread/block indices to the data index (i)?";
    auto response_01 =
        "(C): i = blockIdx.x * blockDim.x + threadIdx.x";
    std::cout << exercise_01 << std::endl;
    std::cout << response_01 << std::endl;

    return 0;
}
