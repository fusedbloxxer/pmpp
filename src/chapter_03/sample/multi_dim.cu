#include "multi_dim.h"

__global__ void vecAddKernel()
{
}

int chapter_03::sample::main()
{
    dim3 dimGrid = {32, 1, 1};
    vecAddKernel<<<dimGrid, {128, 1, 1}>>>();

    return 0;
}
