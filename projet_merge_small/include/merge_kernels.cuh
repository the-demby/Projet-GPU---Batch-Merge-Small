// include/merge_kernels.cuh
#ifndef MERGE_KERNELS_CUH
#define MERGE_KERNELS_CUH

__global__ void mergeSmall_k(int *A, int *B, int *M, int sizeA, int sizeB);

#endif

__global__ void mergeSmallBatch_k(int *A, int *B, int *M, int d, int N);

__global__ void mergeSmallBatchShared_k(int *A, int *B, int *M, int d, int N);

__global__ void mergeSmallOnePerBlock_k(int *A, int *B, int *M, int d, int N);

__global__ void mergeSmallCoopWarp_k(int *A, int *B, int *M, int d, int N);
