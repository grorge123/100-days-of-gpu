#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int CONVERT_BLOCK_X = 16;
static constexpr int CONVERT_BLOCK_Y = 16;
static constexpr int EDGE_BLOCK_X    = 16;
static constexpr int EDGE_BLOCK_Y    = 16;

__global__ void fp32_to_half_2d(const float* __restrict__ src,
                                int src_ld,
                                half* __restrict__ dst,
                                int dst_ld,
                                int rows,
                                int cols) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        dst[static_cast<size_t>(r) * dst_ld + c] = __float2half(src[static_cast<size_t>(r) * src_ld + c]);
    }
}

__global__ void gemm_scalar_rect(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M,
                                 int N,
                                 int K,
                                 int rowBegin,
                                 int rowEnd,
                                 int colBegin,
                                 int colEnd) {
    const int col = colBegin + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = rowBegin + blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rowEnd || col >= colEnd) return;

    float acc = 0.0f;
    const float* aRow = A + static_cast<size_t>(row) * N;
#pragma unroll 4
    for (int k = 0; k < N; ++k) {
        acc = fmaf(aRow[k], B[static_cast<size_t>(k) * K + col], acc);
    }
    C[static_cast<size_t>(row) * K + col] = acc;
}

__global__ void gemm_tail_add_rect(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N,
                                   int K,
                                   int fullM,
                                   int fullK,
                                   int tailStart) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= fullM || col >= fullK) return;

    float acc = C[static_cast<size_t>(row) * K + col];
    const float* aRow = A + static_cast<size_t>(row) * N;
#pragma unroll 4
    for (int k = tailStart; k < N; ++k) {
        acc = fmaf(aRow[k], B[static_cast<size_t>(k) * K + col], acc);
    }
    C[static_cast<size_t>(row) * K + col] = acc;
}

__global__ void gemm_wmma_correction_rect(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          const half* __restrict__ A_half,
                                          const half* __restrict__ B_half,
                                          float* __restrict__ C,
                                          int N,
                                          int K,
                                          int fullM,
                                          int fullN,
                                          int fullK) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= fullM || col >= fullK) return;

    float acc = C[static_cast<size_t>(row) * K + col];
    const float* aRow = A + static_cast<size_t>(row) * N;
    const half* aHalfRow = A_half + static_cast<size_t>(row) * fullN;

#pragma unroll 4
    for (int k = 0; k < fullN; ++k) {
        const float a_half = __half2float(aHalfRow[k]);
        const size_t bIdx = static_cast<size_t>(k) * K + col;
        const float b_full = B[bIdx];
        const float b_half = __half2float(B_half[bIdx]);

        acc = fmaf(aRow[k] - a_half, b_full, acc);
        acc = fmaf(a_half, b_full - b_half, acc);
    }

    C[static_cast<size_t>(row) * K + col] = acc;
}

// Correctness-focused WMMA kernel.
// One warp computes one 16x16 tile, but A/B are staged into shared memory with
// a fixed leading dimension of 16 so load_matrix_sync never depends on arbitrary
// global leading dimensions. The result is also stored to shared memory first,
// then scattered to global C, so K does not need to satisfy WMMA store alignment.
__global__ void gemm_wmma_16x16_fulltiles_fixed(const half* __restrict__ A,
                                                const half* __restrict__ B,
                                                float* __restrict__ C,
                                                int Nfull,
                                                int K) {
    const int lane = threadIdx.x & 31;
    const int tileRow = blockIdx.y * WMMA_M;
    const int tileCol = blockIdx.x * WMMA_N;

    __shared__ __align__(32) half sA[WMMA_M * WMMA_K];
    __shared__ __align__(32) half sB[WMMA_K * WMMA_N];
    __shared__ __align__(32) float sC[WMMA_M * WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int kb = 0; kb < Nfull; kb += WMMA_K) {
        for (int idx = lane; idx < WMMA_M * WMMA_K; idx += 32) {
            const int r = idx / WMMA_K;
            const int c = idx % WMMA_K;
            sA[idx] = A[(static_cast<size_t>(tileRow + r) * Nfull) + (kb + c)];
            sB[idx] = B[(static_cast<size_t>(kb + r) * K) + (tileCol + c)];
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, sA, WMMA_K);
        wmma::load_matrix_sync(b_frag, sB, WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(sC, c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
        const int r = idx / WMMA_N;
        const int c = idx % WMMA_N;
        C[(static_cast<size_t>(tileRow + r) * K) + (tileCol + c)] = sC[idx];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    const bool wmma_supported = (prop.major >= 7);
    const bool apply_correction = true;
    const int fullM = (M / WMMA_M) * WMMA_M;
    const int fullN = (N / WMMA_K) * WMMA_K;
    const int fullK = (K / WMMA_N) * WMMA_N;

    if (!wmma_supported || fullM == 0 || fullN == 0 || fullK == 0) {
        dim3 block(EDGE_BLOCK_X, EDGE_BLOCK_Y);
        dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_scalar_rect<<<grid, block>>>(A, B, C, M, N, K, 0, M, 0, K);
        cudaDeviceSynchronize();
        return;
    }

    half* A_half = nullptr;
    half* B_half = nullptr;
    cudaMalloc(&A_half, static_cast<size_t>(M) * static_cast<size_t>(fullN) * sizeof(half));
    cudaMalloc(&B_half, static_cast<size_t>(fullN) * static_cast<size_t>(K) * sizeof(half));

    {
        dim3 block(CONVERT_BLOCK_X, CONVERT_BLOCK_Y);
        dim3 gridA((fullN + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        dim3 gridB((K + block.x - 1) / block.x, (fullN + block.y - 1) / block.y);
        fp32_to_half_2d<<<gridA, block>>>(A, N, A_half, fullN, M, fullN);
        fp32_to_half_2d<<<gridB, block>>>(B, K, B_half, K, fullN, K);
    }

    {
        dim3 block(32);
        dim3 grid(fullK / WMMA_N, fullM / WMMA_M);
        gemm_wmma_16x16_fulltiles_fixed<<<grid, block>>>(A_half, B_half, C, fullN, K);
    }

    {
        dim3 block(EDGE_BLOCK_X, EDGE_BLOCK_Y);
        dim3 grid((fullK + block.x - 1) / block.x, (fullM + block.y - 1) / block.y);
        gemm_wmma_correction_rect<<<grid, block>>>(A, B, A_half, B_half, C, N, K, fullM, fullN, fullK);
    }

    if (fullN < N) {
        dim3 block(EDGE_BLOCK_X, EDGE_BLOCK_Y);
        dim3 grid((fullK + block.x - 1) / block.x, (fullM + block.y - 1) / block.y);
        gemm_tail_add_rect<<<grid, block>>>(A, B, C, N, K, fullM, fullK, fullN);
    }

    if (fullK < K) {
        dim3 block(EDGE_BLOCK_X, EDGE_BLOCK_Y);
        dim3 grid((K - fullK + block.x - 1) / block.x,
                  (fullM + block.y - 1) / block.y);
        gemm_scalar_rect<<<grid, block>>>(A, B, C, M, N, K, 0, fullM, fullK, K);
    }

    if (fullM < M) {
        dim3 block(EDGE_BLOCK_X, EDGE_BLOCK_Y);
        dim3 grid((K + block.x - 1) / block.x,
                  (M - fullM + block.y - 1) / block.y);
        gemm_scalar_rect<<<grid, block>>>(A, B, C, M, N, K, fullM, M, 0, K);
    }

    cudaFree(A_half);
    cudaFree(B_half);
    cudaDeviceSynchronize();
}