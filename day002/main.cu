#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K);

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(_e));      \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

static void cpu_gemm_ref(const float* A, const float* B, float* Cref,
                         int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int j = 0; j < N; ++j) {
                acc += static_cast<double>(A[i * N + j]) *
                       static_cast<double>(B[j * K + k]);
            }
            Cref[i * K + k] = static_cast<float>(acc);
        }
    }
}

static double ms_since(std::chrono::high_resolution_clock::time_point t0,
                       std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static bool parse_verify_flag(int argc, char** argv, bool default_value) {
    bool verify = default_value;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verify") verify = true;
        if (std::string(argv[i]) == "--no-verify") verify = false;
    }
    return verify;
}

static void parse_positional(int argc, char** argv, int& M, int& N, int& K, int& iters) {
    std::vector<int> nums;
    nums.reserve(4);
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-' && argv[i][1] == '-') continue;
        char* endp = nullptr;
        long v = std::strtol(argv[i], &endp, 10);
        if (endp != argv[i] && *endp == '\0') nums.push_back(static_cast<int>(v));
    }
    if (nums.size() >= 3) { M = nums[0]; N = nums[1]; K = nums[2]; }
    if (nums.size() >= 4) { iters = std::max(1, nums[3]); }
}

int main(int argc, char** argv) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Warp size: %d\n", prop.warpSize);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    int M = 8192, N = 6144, K = 4096;
    int iters = 20;
    int warmup = 3;

    bool verify = parse_verify_flag(argc, argv, true);
    parse_positional(argc, argv, M, N, K, iters);

    std::printf("M=%d N=%d K=%d, warmup=%d, iters=%d, verify=%s\n",
                M, N, K, warmup, iters, verify ? "ON" : "OFF");

    const size_t bytesA = static_cast<size_t>(M) * N * sizeof(float);
    const size_t bytesB = static_cast<size_t>(N) * K * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * K * sizeof(float);

    std::vector<float> hA(static_cast<size_t>(M) * N);
    std::vector<float> hB(static_cast<size_t>(N) * K);
    std::vector<float> hC(static_cast<size_t>(M) * K, 0.0f);

    std::vector<float> hCref;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(evStart));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, evStart, evStop));

    for (int i = 0; i < warmup; ++i) solve(dA, dB, dC, M, N, K);

    double kernel_sum_ms = 0.0;
    double kernel_min_ms = 1e100;
    double kernel_max_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));

        CUDA_CHECK(cudaEventRecord(evStart));
        solve(dA, dB, dC, M, N, K); 
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));

        kernel_sum_ms += ms;
        kernel_min_ms = std::min(kernel_min_ms, static_cast<double>(ms));
        kernel_max_ms = std::max(kernel_max_ms, static_cast<double>(ms));
    }

    const double kernel_avg_ms = kernel_sum_ms / iters;

    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(evStart));
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, evStart, evStop));

    const double gpu_end2end_ms =
        static_cast<double>(h2d_ms) + kernel_avg_ms + static_cast<double>(d2h_ms);

    std::printf("\nTiming (ms):\n");
    std::printf("  H2D copy:          %.3f\n", h2d_ms);
    std::printf("  Kernel avg:        %.3f  (min %.3f, max %.3f) over %d iters\n",
                kernel_avg_ms, kernel_min_ms, kernel_max_ms, iters);
    std::printf("  D2H copy:          %.3f\n", d2h_ms);
    std::printf("  GPU end-to-end:    %.3f  (H2D + Kernel(avg) + D2H)\n", gpu_end2end_ms);


    if (verify) {
        hCref.assign(static_cast<size_t>(M) * K, 0.0f);

        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        cpu_gemm_ref(hA.data(), hB.data(), hCref.data(), M, N, K);
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = ms_since(cpu_t0, cpu_t1);

        const float atol = 1e-3f;
        const float rtol = 1e-3f;
        float maxAbsErr = 0.0f;
        float maxRelErr = 0.0f;
        int badCount = 0;

        for (int i = 0; i < M * K; ++i) {
            float ref = hCref[i];
            float got = hC[i];
            float absErr = std::fabs(got - ref);
            float relErr = absErr / (std::fabs(ref) + 1e-6f);

            maxAbsErr = std::max(maxAbsErr, absErr);
            maxRelErr = std::max(maxRelErr, relErr);

            if (!(absErr <= atol || relErr <= rtol)) {
                if (badCount < 10) {
                    std::printf("Mismatch at idx %d: got=%f ref=%f abs=%f rel=%f\n",
                                i, got, ref, absErr, relErr);
                }
                ++badCount;
            }
        }

        std::printf("\nCPU reference:\n");
        std::printf("  CPU time:          %.3f ms\n", cpu_ms);
        std::printf("  maxAbsErr=%g maxRelErr=%g badCount=%d\n", maxAbsErr, maxRelErr, badCount);
        std::printf("  %s\n", (badCount == 0) ? "PASS" : "FAIL");

        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));

        return (badCount == 0) ? 0 : 1;
    }

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
