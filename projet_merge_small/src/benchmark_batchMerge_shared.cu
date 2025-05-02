#include <stdio.h>
#include <cuda.h>
#include "../include/merge_kernels.cuh"

void benchmark_mergeBatchShared(int d, int N) {
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
    int numThreadsPerMerge = d;

    int sharedMemMaxBytes = 48 * 1024; // 48 KB
    int mergesPerBlock = min(threadsPerBlock / numThreadsPerMerge, (int)(sharedMemMaxBytes / (d * sizeof(int))));

    if (mergesPerBlock == 0) {
        printf("[Shared] d = %4d | ❌ Pas assez de mémoire partagée pour 1 fusion par bloc.\n", d);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
        delete[] h_A; delete[] h_B; delete[] h_M;
        return;
    }

    int numBlocks = (N + mergesPerBlock - 1) / mergesPerBlock;
    size_t sharedMemSize = mergesPerBlock * d * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSmallBatchShared_k<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_M, d, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("[Shared] d = %4d | N = %6d | Time = %7.3f ms | merges/block = %d\n",
           d, N, milliseconds, mergesPerBlock);

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
        benchmark_mergeBatchShared(d_values[i], N);
    }

    return 0;
}
