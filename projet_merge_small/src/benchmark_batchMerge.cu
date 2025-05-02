// src/benchmark_batchMerge.cu
#include <stdio.h>
#include <cuda.h>
#include "../include/merge_kernels.cuh"

void benchmark_mergeBatch(int d, int N) {
    int sizeA = d / 2;
    int totalA = N * sizeA;
    int totalM = N * d;

    int *h_A = new int[totalA];
    int *h_B = new int[totalA];
    int *h_M = new int[totalM];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < sizeA; ++j) {
            h_A[i * sizeA + j] = 2 * j;
            h_B[i * sizeA + j] = 2 * j + 1;
        }
    }

    int *d_A, *d_B, *d_M;
    cudaMalloc(&d_A, totalA * sizeof(int));
    cudaMalloc(&d_B, totalA * sizeof(int));
    cudaMalloc(&d_M, totalM * sizeof(int));

    cudaMemcpy(d_A, h_A, totalA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, totalA * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N * d + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_M, d, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("d = %4d | N = %6d | Time = %7.3f ms\n", d, N, milliseconds);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);
    delete[] h_A;
    delete[] h_B;
    delete[] h_M;
}

int main() {
    int d_values[] = {128, 256, 512, 1024};
    const int N = 10000;

    for (int i = 0; i < 4; ++i) {
        benchmark_mergeBatch(d_values[i], N);
    }

    return 0;
}
