#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  int N = 10;
  std::cin >> N;
  std::vector<float> A(N), B(N), C(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 100.0f);

  for (int i = 0; i < N; i++) {
    A[i] = dis(gen);
    B[i] = dis(gen);
  }
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_c, N * sizeof(float));
  cudaMemcpy(d_a, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  int blocksize = 256;
  int gridsize = (N + blocksize - 1) / blocksize;
  vecAdd<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  cudaMemcpy(C.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}