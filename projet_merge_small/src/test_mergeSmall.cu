// src/test_mergeSmall.cu
#include <stdio.h>
#include <cuda.h>
#include "../include/merge_kernels.cuh"

void test_mergeSmall() {
    const int sizeA = 5;
    const int sizeB = 5;
    const int sizeM = sizeA + sizeB;

    int h_A[sizeA] = {1, 3, 5, 7, 9};
    int h_B[sizeB] = {2, 4, 6, 8, 10};
    int h_M[sizeM];

    int *d_A, *d_B, *d_M;
    cudaMalloc(&d_A, sizeA * sizeof(int));
    cudaMalloc(&d_B, sizeB * sizeof(int));
    cudaMalloc(&d_M, sizeM * sizeof(int));

    cudaMemcpy(d_A, h_A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(int), cudaMemcpyHostToDevice);

    mergeSmall_k<<<1, sizeM>>>(d_A, d_B, d_M, sizeA, sizeB);
    cudaDeviceSynchronize();

    cudaMemcpy(h_M, d_M, sizeM * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Résultat de la fusion A + B :\n");
    for (int i = 0; i < sizeM; ++i) {
        printf("%d ", h_M[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);
}

int main() {
    test_mergeSmall();
    return 0;
}
