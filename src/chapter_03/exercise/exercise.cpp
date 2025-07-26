#include <cassert>

#include "exercise.h"

#include "mm.h"

int chapter_03::exercise::main()
{
    torch::Tensor a = torch::randn({4, 4}).contiguous();
    torch::Tensor b = torch::randn({4, 4}).contiguous();
    torch::Tensor v = torch::randn({4}).contiguous();

    // Exercise 1.a
    torch::Tensor c_row = mm::mm_row(a, b);
    assert(torch::allclose(c_row, torch::matmul(a, b), 0, 1e-5));

    // Exercise 1.b
    torch::Tensor c_col = mm::mm_col(a, b);
    assert(torch::allclose(c_col, torch::matmul(a, b), 0, 1e-5));

    // Exercise 2
    torch::Tensor c_v = mm::mv(a, v);
    assert(torch::allclose(c_v, torch::matmul(a, v), 0, 1e-5));

    // Exercise 3
    std::cout << "3.a -  32x16  =   512 threads per block" << std::endl;
    std::cout << "3.b -   5x19  =    95 blocks in grid" << std::endl;
    std::cout << "3.c -  95x512 = 48640 threads in the grid" << std::endl;
    std::cout << "3.d - 150x300 = 45000 threads execute line 05" << std::endl;

    // Exercise 4
    std::cout << "4.a i = 20 * 400 + 10 = 8010" << std::endl;
    std::cout << "4.b i = 10 * 500 + 20 = 5020" << std::endl;

    // Exercise 5
    std::cout << "5 - i = 5 * 500 * 400 + 20 * 400 + 10 = 1008010" << std::endl;

    return 0;
}
