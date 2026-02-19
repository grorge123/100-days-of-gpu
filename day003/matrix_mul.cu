// gemm_optimized.cu
// A: MxN, B: NxK, C: MxK (row-major). Pointers are DEVICE pointers.
// Fast path (cp.async + vectorized shared/global ops) is used when N%4==0 and K%4==0.
//
// Build (RTX 3070 = sm_86):
//   nvcc -O3 -lineinfo -arch=sm_86 gemm_optimized.cu -c -o gemm_optimized.o
//
// Notes:
// - Uses cp.async.cg (L2-prefer) on Ampere+
// - Shared padding uses PAD=4 (keeps 16B alignment for cp.async 16B and reduces bank conflicts)
// - Uses float2 loads from shared B inside compute loop to reduce LSU/MIO pressure
// - Uses float2 stores to C where possible to reduce store instructions
// - Keeps a scalar fallback kernel for general (unaligned / edge-heavy) cases

#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static constexpr int BLOCK_X = 16;
static constexpr int BLOCK_Y = 16;

static constexpr int TILE_M = 32; // C tile rows per block
static constexpr int TILE_K = 32; // C tile cols per block
static constexpr int TILE_N = 32; // reduction tile

static constexpr int PAD = 4;     // must be multiple of 4 floats => 16B alignment preserved

// ------------------------ cp.async helpers (Ampere+) ------------------------
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_cg_16B(void* smem_dst, const void* gmem_src) {
    unsigned int smem_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

// ------------------------ Fast kernel: 2x2 register tile + float4 cp.async + float2 shared load/store ------------------------
__global__ void gemm_rt2x2_v4_cpasync(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15

    const int blockRow = blockIdx.y * TILE_M;
    const int blockCol = blockIdx.x * TILE_K;

    // output indices (each thread computes 2x2)
    const int r0 = blockRow + (ty * 2 + 0);
    const int r1 = blockRow + (ty * 2 + 1);
    const int c0 = blockCol + (tx * 2 + 0);
    const int c1 = blockCol + (tx * 2 + 1);

    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;

    __shared__ __align__(16) float As[2][TILE_M][TILE_N + PAD];   // 32 x 36
    __shared__ __align__(16) float Bs[2][TILE_N][TILE_K + PAD];   // 32 x 36

    const int numTiles = (N + TILE_N - 1) / TILE_N;

    // cooperative load mapping: each thread copies 16B (float4) for A and 16B for B
    const int local_row_sel = (tx >> 3);   // 0 or 1
    const int local_seg     = (tx & 7);    // 0..7
    const int a_row_local   = ty * 2 + local_row_sel; // 0..31
    const int k_base        = local_seg * 4;           // 0,4,...,28

    const int b_k_local     = ty * 2 + local_row_sel;  // 0..31
    const int c_base        = local_seg * 4;           // 0,4,...,28

    auto prefetch_tile = [&](int tIdx, int st) {
        const int tileN = tIdx * TILE_N;

        const bool full_tile =
            (blockRow + TILE_M - 1 < M) &&
            (blockCol + TILE_K - 1 < K) &&
            (tileN   + TILE_N - 1 < N);

#if __CUDA_ARCH__ >= 800
        if (full_tile) {
            // A -> As
            const int aRow = blockRow + a_row_local;
            const float* gA = A + aRow * N + (tileN + k_base);
            float* sA = &As[st][a_row_local][k_base];
            cp_async_cg_16B(sA, gA);

            // B -> Bs
            const int bRow = tileN + b_k_local;
            const float* gB = B + bRow * K + (blockCol + c_base);
            float* sB = &Bs[st][b_k_local][c_base];
            cp_async_cg_16B(sB, gB);
        } else
#endif
        {
            // safe scalar fallback for boundary tiles
            const int aRow = blockRow + a_row_local;
            const int aColBase = tileN + k_base;
            float* sA = &As[st][a_row_local][k_base];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int aCol = aColBase + i;
                sA[i] = (aRow < M && aCol < N) ? A[aRow * N + aCol] : 0.0f;
            }

            const int bRow = tileN + b_k_local;
            const int bColBase = blockCol + c_base;
            float* sB = &Bs[st][b_k_local][c_base];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int bCol = bColBase + i;
                sB[i] = (bRow < N && bCol < K) ? B[bRow * K + bCol] : 0.0f;
            }
        }
    };

    int stage = 0;
    prefetch_tile(0, stage);

#if __CUDA_ARCH__ >= 800
    cp_async_commit();
    cp_async_wait_all();
#endif
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        const int next_t = t + 1;
        const int next_stage = stage ^ 1;

        if (next_t < numTiles) {
            prefetch_tile(next_t, next_stage);
#if __CUDA_ARCH__ >= 800
            cp_async_commit();
#endif
        }

        const int r0_local = ty * 2 + 0;
        const int r1_local = ty * 2 + 1;
        const int c0_local = tx * 2 + 0; // even
        // c1_local = c0_local + 1

#pragma unroll
        for (int k = 0; k < TILE_N; ++k) {
            const float a0 = As[stage][r0_local][k];
            const float a1 = As[stage][r1_local][k];

            // Reduce LSU/MIO pressure: load two B values with one instruction
            const float2 b = *reinterpret_cast<const float2*>(&Bs[stage][k][c0_local]);

            acc00 = fmaf(a0, b.x, acc00);
            acc01 = fmaf(a0, b.y, acc01);
            acc10 = fmaf(a1, b.x, acc10);
            acc11 = fmaf(a1, b.y, acc11);
        }

        if (next_t < numTiles) {
#if __CUDA_ARCH__ >= 800
            cp_async_wait_all();
#endif
            __syncthreads();
            stage = next_stage;
        }
    }

    // Stores: use float2 when possible (reduces store instructions)
    if (r0 < M) {
        if (c1 < K) {
            *reinterpret_cast<float2*>(C + r0 * K + c0) = make_float2(acc00, acc01);
        } else if (c0 < K) {
            C[r0 * K + c0] = acc00;
        }
    }
    if (r1 < M) {
        if (c1 < K) {
            *reinterpret_cast<float2*>(C + r1 * K + c0) = make_float2(acc10, acc11);
        } else if (c0 < K) {
            C[r1 * K + c0] = acc10;
        }
    }
}

// ------------------------ Scalar fallback kernel (no cp.async assumptions) ------------------------
__global__ void gemm_rt2x2_scalar(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int blockRow = blockIdx.y * TILE_M;
    const int blockCol = blockIdx.x * TILE_K;

    const int r0 = blockRow + (ty * 2 + 0);
    const int r1 = blockRow + (ty * 2 + 1);
    const int c0 = blockCol + (tx * 2 + 0);
    const int c1 = blockCol + (tx * 2 + 1);

    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;

    __shared__ float As[TILE_M][TILE_N + 1];       // padding to reduce bank conflicts
    __shared__ float Bs[TILE_N][TILE_K + 1];

    const int numTiles = (N + TILE_N - 1) / TILE_N;

    const int local_row_sel = (tx >> 3);
    const int local_seg     = (tx & 7);
    const int a_row_local   = ty * 2 + local_row_sel;
    const int k_base        = local_seg * 4;

    const int b_k_local     = ty * 2 + local_row_sel;
    const int c_base        = local_seg * 4;

    for (int t = 0; t < numTiles; ++t) {
        const int tileN = t * TILE_N;

        // load A (4 floats)
        {
            const int aRow = blockRow + a_row_local;
            const int aColBase = tileN + k_base;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int aCol = aColBase + i;
                As[a_row_local][k_base + i] = (aRow < M && aCol < N) ? A[aRow * N + aCol] : 0.0f;
            }
        }
        // load B (4 floats)
        {
            const int bRow = tileN + b_k_local;
            const int bColBase = blockCol + c_base;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int bCol = bColBase + i;
                Bs[b_k_local][c_base + i] = (bRow < N && bCol < K) ? B[bRow * K + bCol] : 0.0f;
            }
        }

        __syncthreads();

        const int r0_local = ty * 2 + 0;
        const int r1_local = ty * 2 + 1;
        const int c0_local = tx * 2 + 0;

#pragma unroll
        for (int k = 0; k < TILE_N; ++k) {
            const float a0 = As[r0_local][k];
            const float a1 = As[r1_local][k];
            const float2 b = *reinterpret_cast<const float2*>(&Bs[k][c0_local]);

            acc00 = fmaf(a0, b.x, acc00);
            acc01 = fmaf(a0, b.y, acc01);
            acc10 = fmaf(a1, b.x, acc10);
            acc11 = fmaf(a1, b.y, acc11);
        }

        __syncthreads();
    }

    if (r0 < M && c0 < K) C[r0 * K + c0] = acc00;
    if (r0 < M && c1 < K) C[r0 * K + c1] = acc01;
    if (r1 < M && c0 < K) C[r1 * K + c0] = acc10;
    if (r1 < M && c1 < K) C[r1 * K + c1] = acc11;
}

// ------------------------ Host entry ------------------------
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((K + TILE_K - 1) / TILE_K,
              (M + TILE_M - 1) / TILE_M,
              1);

    // fast path requires vector-friendly leading dimensions for 16B/8B alignment assumptions
    const bool aligned_fast = ((N & 3) == 0) && ((K & 3) == 0);

    if (aligned_fast) {
        gemm_rt2x2_v4_cpasync<<<grid, block>>>(A, B, C, M, N, K);
    } else {
        gemm_rt2x2_scalar<<<grid, block>>>(A, B, C, M, N, K);
    }

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return;

    // Keep sync for correctness given this solve() signature.
    // Remove if caller handles synchronization later.
    (void)cudaDeviceSynchronize();
}
