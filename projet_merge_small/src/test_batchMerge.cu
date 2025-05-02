// src/test_batchMerge.cu
#include <stdio.h>
#include <cuda.h>
#include "../include/merge_kernels.cuh"

void test_mergeSmallBatch() {
    const int d = 8;
    const int sizeA = d / 2;
    const int sizeB = d / 2;
    const int N = 3;

    int h_A[N * sizeA] = {
        1, 3, 5, 7,
        2, 4, 6, 8,
        0, 9, 10, 11
    };

    int h_B[N * sizeB] = {
        2, 6, 8, 10,
        1, 5, 7, 9,
        3, 4, 12, 13
    };

    int h_M[N * d];

    int *d_A, *d_B, *d_M;
    cudaMalloc(&d_A, N * sizeA * sizeof(int));
    cudaMalloc(&d_B, N * sizeB * sizeof(int));
    cudaMalloc(&d_M, N * d * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeB * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 32; // multiple de d
    int numBlocks = (N * d + threadsPerBlock - 1) / threadsPerBlock;

    mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_M, d, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_M, d_M, N * d * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Résultat batch fusion :\n");
    for (int i = 0; i < N; ++i) {
        printf("Couple %d : ", i);
        for (int j = 0; j < d; ++j) {
            printf("%d ", h_M[i * d + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);
}

int main() {
    test_mergeSmallBatch();
    return 0;
}
