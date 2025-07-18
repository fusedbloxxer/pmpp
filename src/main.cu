#include <iostream>

__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n)
{
    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    size_t bytes = n * sizeof(*A_d);

    // Allocate memory on device
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    // Host -> Device
    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    // Device -> Host
    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return;
}

int main()
{
    float A[5] = {1, 2, 3, 4, 5};
    float B[5] = {9, 8, 6, 6, 5};
    float C[5] = {0};

    vecAdd(A, B, C, sizeof(A) / sizeof(*A));

    for (int i = 0; i != 5; i++)
    {
        std::cout << C[i] << " " << std::endl;
    }

    return 0;
}
