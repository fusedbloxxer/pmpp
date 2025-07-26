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
    std::cout << std::endl;

    auto exercise_02 =
        "2. . Assume that we want to use each thread to calculate two adjacent elements of a vector addition."
        "What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?";
    auto response_02 =
        "(C) i=(blockIdx.x blockDim.x + threadIdx.x) 2";
    std::cout << exercise_02 << std::endl;
    std::cout << response_02 << std::endl;
    std::cout << std::endl;

    auto exercise_03 =
        "3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes"
        "2 blockDim.x consecutive elements that form two sections. All threads in each block will process a section"
        " first, each processing one element. They will then all move to the next section, each processing one element."
        "Assume that variable i should be the index for the first element to be processed by a thread."
        "What would be the expression for mapping the indices to data index of the first element? ";
    auto response_03 =
        "(D) i=blockIdx.x blockDim.x 2 + threadIdx.x";
    std::cout << exercise_03 << std::endl;
    std::cout << response_03 << std::endl;
    std::cout << std::endl;

    auto exercise_04 =
        "4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element,"
        "and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number"
        "of thread blocks to cover all output elements. How many threads will be in the grid?";
    auto response_04 =
        "(C) 8192";
    std::cout << exercise_04 << std::endl;
    std::cout << response_04 << std::endl;
    std::cout << std::endl;

    auto exercise_05 =
        ". 5. If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an"
        "appropriate expression for the second argument of the cudaMalloc call? ";
    auto response_05 =
        "(D) sizeof(int)*v";
    std::cout << exercise_05 << std::endl;
    std::cout << response_05 << std::endl;
    std::cout << std::endl;

    auto exercise_06 =
        "6. If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d"
        "to point to the allocated memory, what would be an appropriate expression for the first argument of the cudaMalloc () call?";
    auto response_06 =
        "(d) (void**)&A_d";
    std::cout << exercise_06 << std::endl;
    std::cout << response_06 << std::endl;
    std::cout << std::endl;

    auto exercise_07 =
        "7. If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array)"
        "to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA? ";
    auto response_07 =
        "(C) cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);";
    std::cout << exercise_07 << std::endl;
    std::cout << response_07 << std::endl;
    std::cout << std::endl;

    auto exercise_08 = "8. How would one declare a variable err that can appropriately receive"
                       "the returned value of a CUDA API call?";
    auto response_08 = "(C) cudaError_t err; or (B) cudaError err";
    std::cout << exercise_08 << std::endl;
    std::cout << response_08 << std::endl;
    std::cout << std::endl;

    auto exercise_09 = "9. Consider the following CUDA kernel and the corresponding host function that calls it: ";
    std::cout << exercise_09 << std::endl;
    std::cout << "a. 128" << std::endl;
    std::cout << "b. 200064" << std::endl;
    std::cout << "c. 1563" << std::endl;
    std::cout << "d. 200064" << std::endl;
    std::cout << "e. 200000" << std::endl;
    std::cout << std::endl;

    return 0;
}
