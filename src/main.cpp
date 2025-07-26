#include <filesystem>
#include <iostream>
#include <torch/torch.h>

#include "./chapter_02/chapter_02.h"
#include "./chapter_03/chapter_03.h"
#include "./shared/shared.h"

namespace fs = std::filesystem;

int main()
{
    torch::Tensor tensor = torch::rand({2, 3}).to(torch::kCUDA);

    std::cout << tensor << std::endl;

    return 0;
}
