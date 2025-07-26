#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../../shared/shared.h"
#include "blur_kernel.h"

__global__ void blur_kernel(uchar *inp, uchar *out, int c, int h, int w, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int chn = blockIdx.z;

    if (row >= h || col >= w)
    {
        return;
    }

    int pixel_count = 0;
    int pixel_value = 0;

    for (int row_b = -k; row_b != k; ++row_b)
    {
        for (int col_b = -k; col_b != k; ++col_b)
        {
            int curRow = row + row_b;
            int curCol = col + col_b;

            if (curRow < 0 || curRow >= h || curCol < 0 || curCol >= w)
            {
                continue;
            }

            pixel_count++;
            pixel_value += inp[c * (curRow * w + curCol) + chn];
        }
    }

    out[c * (row * w + col) + chn] = (uchar)(pixel_value / pixel_count);
}

void image_blur(cv::Mat &img, const int kernel_size)
{
    uchar *Img_inp = nullptr;
    uchar *Img_out = nullptr;

    uint c = img.channels();
    uint h = img.rows;
    uint w = img.cols;
    uint bytes = c * h * w * sizeof(uchar);

    cudaMalloc(&Img_inp, bytes);
    cudaMalloc(&Img_out, bytes);

    cudaMemcpy(Img_inp, img.datastart, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    dim3 gridDim{static_cast<uint>(ceil(w / 16)), static_cast<uint>(ceil(h / 16)), c};
    dim3 blockDim{16, 16, 1};
    blur_kernel<<<gridDim, blockDim>>>(Img_inp, Img_out, c, h, w, kernel_size);
    cudaMemcpy((void *)img.datastart, Img_out, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(Img_inp);
    cudaFree(Img_out);
}

int chapter_03::sample::blur_kernel::main()
{
    const auto &res = shared::globals::get_instance()->get_res();
    const auto &inp = res / "cityscape.png";
    const auto &out = res / "cityscape_blurred.png";

    cv::Mat img_inp = cv::imread(inp, cv::ImreadModes::IMREAD_COLOR_BGR);
    cv::imshow("Input Image", img_inp);
    cv::waitKey(0);

    cv::Mat img_out = img_inp.clone();
    cv::cvtColor(img_out, img_out, cv::ColorConversionCodes::COLOR_BGR2RGB);

    image_blur(img_out, 7);

    cv::cvtColor(img_out, img_out, cv::ColorConversionCodes::COLOR_RGB2BGR);
    cv::imshow("Output Image", img_out);
    cv::waitKey(0);

    cv::imwrite(out, img_out);

    return 0;
}
