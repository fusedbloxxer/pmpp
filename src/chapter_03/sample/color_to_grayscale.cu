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

__global__ void colorToGrayscaleConversion(unsigned char *Pin_d, unsigned char *Pout_d, int channels, int height, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width)
    {
        return;
    }

    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * channels;

    unsigned char r = Pin_d[rgbOffset];
    unsigned char g = Pin_d[rgbOffset + 1];
    unsigned char b = Pin_d[rgbOffset + 2];

    Pout_d[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
}

int chapter_03::sample::main()
{
    const auto &cwd = shared::globals::get_instance()->get_res();
    const auto &img_path_gray = cwd / "cityscape_gray.png";
    const auto &img_path = cwd / "cityscape.png";

    std::cout << "Reading the image at: " << img_path << std::endl;
    Mat img = imread(img_path, IMREAD_COLOR);

    if (img.empty())
    {
        std::cout << "Could not read image: " << img_path << std::endl;
        throw std::runtime_error(std::format("Could not read image: {}!", img_path.string()));
    }

    cvtColor(img, img, COLOR_BGR2RGB);
    size_t channels = img.channels();
    size_t height = img.rows;
    size_t width = img.cols;

    size_t src_size = height * width * img.elemSize();
    size_t dst_size = height * width;

    unsigned char *Pout_h = (unsigned char *)malloc(dst_size);
    unsigned char *Pout_d = nullptr;
    unsigned char *Pin_d = nullptr;

    cudaMalloc(&Pout_d, dst_size);
    cudaMalloc(&Pin_d, src_size);
    cudaMemcpy(Pin_d, img.datastart, src_size, cudaMemcpyHostToDevice);

    uint gridY = ceil(height / 16.0);
    uint gridX = ceil(width / 16.0);
    dim3 gridDim{gridX, gridY, 1};
    dim3 blockDim{16, 16, 1};

    std::cout << gridY << " " << gridX << "test" << std::endl;
    colorToGrayscaleConversion<<<gridDim, blockDim>>>(Pin_d, Pout_d, channels, height, width);

    cudaMemcpy(Pout_h, Pout_d, dst_size, cudaMemcpyDeviceToHost);
    cudaFree(Pout_d);
    cudaFree(Pin_d);

    Mat gray_img(height, width, CV_8UC1, Pout_h);
    imshow("Grayscale Image", gray_img);
    imwrite(img_path_gray, gray_img);
    waitKey(0);

    return 0;
}
