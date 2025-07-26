#include "mm_kernel.h"

__global__ void mm_row_kernel(float *a, float *b, float *c, uint N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
    {
        return;
    }

    for (int c_i = 0; c_i != N; ++c_i)
    {
        float sum = 0;

        for (int k_i = 0; k_i != N; ++k_i)
        {
            sum += a[N * i + k_i] * b[N * k_i + c_i];
        }

        c[N * i + c_i] = sum;
    }
}

__global__ void mm_col_kernel(float *a, float *b, float *c, uint N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
    {
        return;
    }

    for (int r_i = 0; r_i != N; ++r_i)
    {
        float sum = 0;

        for (int k_i = 0; k_i != N; ++k_i)
        {
            sum += a[N * r_i + k_i] * b[N * k_i + i];
        }

        c[N * r_i + i] = sum;
    }
}

__global__ void mv_kernel(float *a, float *b, float *c, uint N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
    {
        return;
    }

    float sum = 0;

    for (int k_i = 0; k_i != N; ++k_i)
    {
        sum += a[N * i + k_i] * b[k_i];
    }

    c[i] = sum;
}

void chapter_03::mm_kernel::mm_row(float *a, float *b, float *c, uint N)
{
    size_t bytes = sizeof(*a) * N * N;
    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;
    uint tpb = 2;

    cudaMalloc(&a_d, bytes);
    cudaMalloc(&b_d, bytes);
    cudaMalloc(&c_d, bytes);

    cudaMemcpy(a_d, a, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

    dim3 blockDim{tpb};
    dim3 gridDim{static_cast<uint>(ceil(N / tpb))};
    mm_row_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, N);

    cudaMemcpy(c, c_d, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void chapter_03::mm_kernel::mm_col(float *a, float *b, float *c, uint N)
{
    size_t bytes = sizeof(*a) * N * N;
    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;
    uint tpb = 2;

    cudaMalloc(&a_d, bytes);
    cudaMalloc(&b_d, bytes);
    cudaMalloc(&c_d, bytes);

    cudaMemcpy(a_d, a, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

    dim3 blockDim{tpb};
    dim3 gridDim{static_cast<uint>(ceil(N / tpb))};
    mm_col_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, N);

    cudaMemcpy(c, c_d, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void chapter_03::mm_kernel::mv(float *a, float *b, float *c, uint N)
{
    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;
    uint tpb = 2;

    cudaMalloc(&a_d, sizeof(*a) * N * N);
    cudaMalloc(&b_d, sizeof(*b) * N);
    cudaMalloc(&c_d, sizeof(*c) * N);

    cudaMemcpy(a_d, a, sizeof(*a) * N * N, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(*b) * N, cudaMemcpyKind::cudaMemcpyHostToDevice);

    dim3 blockDim{tpb};
    dim3 gridDim{static_cast<uint>(ceil(N / tpb))};
    mv_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, N);

    cudaMemcpy(c, c_d, sizeof(*c) * N, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
