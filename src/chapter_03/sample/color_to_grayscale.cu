#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "../../shared/shared.h"
#include "color_to_grayscale.h"

using namespace std::filesystem;
using namespace cv;

__global__ void colorToGrayscaleConversion()
{
}

int chapter_03::sample::main()
{
    const auto &cwd = shared::globals::get_instance()->get_res();
    const auto &img_path = cwd / "cityscape.png";

    std::cout << "Reading the image at: " << img_path << std::endl;
    Mat img = imread(img_path, IMREAD_COLOR);

    if (img.empty())
    {
        std::cout << "Could not read image: " << img_path << std::endl;

        throw std::runtime_error(std::format("Could not read image: {}!", img_path.string()));
    }

    imshow("Display window", img);
    int k = waitKey(0);

    return 0;
}
