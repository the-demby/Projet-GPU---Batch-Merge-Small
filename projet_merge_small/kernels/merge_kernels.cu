// kernels/merge_kernels.cu
#include "../include/merge_kernels.cuh"
#include <cuda.h>

__global__ void mergeSmall_k(int *A, int *B, int *M, int sizeA, int sizeB) {
    int k = threadIdx.x;
    int total = sizeA + sizeB;
    if (k >= total) return;

    int lowA = max(0, k - sizeB);
    int highA = min(k, sizeA);

    while (lowA < highA) {
        int midA = (lowA + highA) / 2;
        int midB = k - midA;
        if (A[midA] < B[midB - 1]) {
            lowA = midA + 1;
        } else {
            highA = midA;
        }
    }

    int i = lowA;
    int j = k - i;

    if (i < sizeA && (j >= sizeB || A[i] <= B[j])) {
        M[k] = A[i];
    } else {
        M[k] = B[j];
    }
}

__global__ void mergeSmallBatch_k(int *A, int *B, int *M, int d, int N) {
    int Qt = threadIdx.x / d;
    int tidx = threadIdx.x % d;
    int gbx = Qt + blockIdx.x * (blockDim.x / d);
    if (gbx >= N || tidx >= d) return;

    int *a = A + gbx * (d / 2);
    int *b = B + gbx * (d / 2);
    int *m = M + gbx * d;

    int sizeA = d / 2;
    int sizeB = d / 2;
    int k = tidx;

    int lowA = max(0, k - sizeB);
    int highA = min(k, sizeA);

    while (lowA < highA) {
        int midA = (lowA + highA) / 2;
        int midB = k - midA;
        if (a[midA] < b[midB - 1]) {
            lowA = midA + 1;
        } else {
            highA = midA;
        }
    }

    int i = lowA;
    int j = k - i;

    if (i < sizeA && (j >= sizeB || a[i] <= b[j])) {
        m[k] = a[i];
    } else {
        m[k] = b[j];
    }
}

__global__ void mergeSmallBatchShared_k(int *A, int *B, int *M, int d, int N) {
    extern __shared__ int shared[]; // espace partagé alloué dynamiquement

    int Qt = threadIdx.x / d;
    int tidx = threadIdx.x % d;
    int gbx = Qt + blockIdx.x * (blockDim.x / d);
    if (gbx >= N || tidx >= d) return;

    int *a = A + gbx * (d / 2);
    int *b = B + gbx * (d / 2);
    int *m = M + gbx * d;

    int *a_shared = shared + Qt * d;
    int *b_shared = a_shared + (d / 2);

    // Chargement des données en mémoire partagée
    if (tidx < d / 2) {
        a_shared[tidx] = a[tidx];
        b_shared[tidx] = b[tidx];
    }
    __syncthreads();

    int k = tidx;
    int sizeA = d / 2;
    int sizeB = d / 2;

    int lowA = max(0, k - sizeB);
    int highA = min(k, sizeA);

    while (lowA < highA) {
        int midA = (lowA + highA) / 2;
        int midB = k - midA;
        if (a_shared[midA] < b_shared[midB - 1]) {
            lowA = midA + 1;
        } else {
            highA = midA;
        }
    }

    int i = lowA;
    int j = k - i;

    if (i < sizeA && (j >= sizeB || a_shared[i] <= b_shared[j])) {
        m[k] = a_shared[i];
    } else {
        m[k] = b_shared[j];
    }
}
__global__ void mergeSmallOnePerBlock_k(int *A, int *B, int *M, int d, int N) {
    int tidx = threadIdx.x;
    int gbx = blockIdx.x;
    if (gbx >= N || tidx >= d) return;

    int *a = A + gbx * (d / 2);
    int *b = B + gbx * (d / 2);
    int *m = M + gbx * d;

    int k = tidx;
    int sizeA = d / 2;
    int sizeB = d / 2;

    int lowA = max(0, k - sizeB);
    int highA = min(k, sizeA);

    while (lowA < highA) {
        int midA = (lowA + highA) / 2;
        int midB = k - midA;
        if (a[midA] < b[midB - 1]) {
            lowA = midA + 1;
        } else {
            highA = midA;
        }
    }

    int i = lowA;
    int j = k - i;

    if (i < sizeA && (j >= sizeB || a[i] <= b[j])) {
        m[k] = a[i];
    } else {
        m[k] = b[j];
    }
}

__global__ void mergeSmallCoopWarp_k(int *A, int *B, int *M, int d, int N) {
    int warpSize = 32;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int warpsPerBlock = blockDim.x / warpSize;

    int gbx = warpId + blockIdx.x * warpsPerBlock;
    if (gbx >= N || laneId >= d) return;

    int *a = A + gbx * (d / 2);
    int *b = B + gbx * (d / 2);
    int *m = M + gbx * d;

    int k = laneId;
    int sizeA = d / 2;
    int sizeB = d / 2;

    int lowA = max(0, k - sizeB);
    int highA = min(k, sizeA);

    while (lowA < highA) {
        int midA = (lowA + highA) / 2;
        int midB = k - midA;
        if (a[midA] < b[midB - 1]) {
            lowA = midA + 1;
        } else {
            highA = midA;
        }
    }

    int i = lowA;
    int j = k - i;

    if (i < sizeA && (j >= sizeB || a[i] <= b[j])) {
        m[k] = a[i];
    } else {
        m[k] = b[j];
    }
}
