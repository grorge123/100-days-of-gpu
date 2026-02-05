#include <cuda_runtime.h>

#ifndef TILE
#define TILE 32          // 32x32 threads = 1024 threads/block
#endif

// A: MxN (row-major)
// B: NxK (row-major)  -> K 維度連續
// C: MxK (row-major)
//
// 讓 block 在 M 方向切（blockIdx.y 對應 row tile），threadIdx.x 對應 K 方向的 col，
// 使得讀 B[j*K + col] 時，col 是連續（coalesced）。
__global__ void matrix_multiplication_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int M, int N, int K) {
    const int row = blockIdx.y * TILE + threadIdx.y; // 沿 M 切
    const int col = blockIdx.x * TILE + threadIdx.x; // 沿 K 切（連續）

    __shared__ float As[TILE][TILE]; // A tile: (rows x Ntile)
    __shared__ float Bs[TILE][TILE]; // B tile: (Ntile x cols)

    float acc = 0.0f;

    // 沿 N（reduction）方向做 tiling
    for (int t = 0; t < N; t += TILE) {
        // Load A tile: A[row, t + x]
        const int a_col = t + threadIdx.x;
        if (row < M && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile: B[t + y, col]
        // 這裡 threadIdx.x 沿 col（K）讀取 => B 的 K 維度連續，coalesced
        const int b_row = t + threadIdx.y;
        if (b_row < N && col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * K + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((K + TILE - 1) / TILE,
                       (M + TILE - 1) / TILE);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
