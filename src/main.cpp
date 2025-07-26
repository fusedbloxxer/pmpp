#include <filesystem>
#include <iostream>

#include "./chapter_02/chapter_02.h"
#include "./chapter_03/chapter_03.h"
#include "./shared/shared.h"

namespace fs = std::filesystem;

int main()
{
    chapter_03::sample::blur_kernel::main();

    return 0;
}
