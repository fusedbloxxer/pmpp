// burn_gpu.cu
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// simple macro to check CUDA calls
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"" << std::endl;                                  \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// kernel: for each element, do a little float arithmetic loop
__global__ void burnKernel(float *data, size_t N, int inner_iters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    float v = data[idx];
    // a small compute loop to keep the ALUs busy
    for (int i = 0; i < inner_iters; ++i)
    {
        v = v * 1.000001f + 0.000001f;
    }
    data[idx] = v;
}

void checkGPUDevices()
{
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    std::cout << "Found " << gpu_count << " CUDA-capable device(s)\n";

    for (int i = 0; i < gpu_count; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout
            << "Device " << i << ": " << prop.name
            << " with " << (prop.totalGlobalMem >> 20) << " MiB\n";
    }
}

int main()
{
    // Check GPUs
    checkGPUDevices();

    // pick GPU #1 (zero‑based). 0 = first card, 1 = second card, etc.
    int device_id = 1;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to set CUDA device #" << device_id
                  << ": " << cudaGetErrorString(err) << "\n";
        return EXIT_FAILURE;
    }

    // 5 GiB of floats
    const size_t bytes = 5ULL * 1024 * 1024 * 1024;
    const size_t N = bytes / sizeof(float);

    std::cout << "Allocating " << bytes / (1024.0 * 1024 * 1024) << " GiB on GPU ("
              << N << " floats)..." << std::endl;

    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // initialize to something
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));

    // launch configuration
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Choose inner_iters to tune compute intensity.
    // On a modern GPU, ~100–500 iters will keep the ALUs busy
    const int inner_iters = 200;

    std::cout << "Launching kernel over " << blocks << " blocks of "
              << threads_per_block << " threads, inner_iters=" << inner_iters
              << std::endl;

    // time measurement
    auto t_start = std::chrono::high_resolution_clock::now();

    // keep launching until 60 seconds have passed
    while (true)
    {
        burnKernel<<<blocks, threads_per_block>>>(d_data, N, inner_iters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t_now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t_now - t_start;
        if (elapsed.count() >= 60.0)
            break;
    }

    std::cout << "Done ~60 seconds of GPU work.\n";
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
