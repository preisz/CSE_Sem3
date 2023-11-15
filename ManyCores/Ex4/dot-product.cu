#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdio.h>


// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[512];

  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    dot += x[i] * y[i];
  }

  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}

int main() {

  Timer timer;
  std::vector<int> Nvals = { 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000, 50'000'000, 100'000'000, };
  std::cout << "****Dot Product****\n";
  std::cout << "Length of vector N, Execution time for operations" << std::endl;

  for (int N : Nvals){
  // Allocate and initialize arrays on CPU
  //
  double *x = (double *)malloc(sizeof(double) * N);
  double *y = (double *)malloc(sizeof(double) * N);
  double alpha = 0;

  std::fill(x, x + N, 1);
  std::fill(y, y + N, 2);

  // Allocate and initialize arrays on GPU
  double *cuda_x;
  double *cuda_y;
  double *cuda_alpha;
  
  CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N));
  CUDA_ERRCHK(cudaMalloc(&cuda_y, sizeof(double) * N));
  CUDA_ERRCHK(cudaMalloc(&cuda_alpha, sizeof(double)));
  
  CUDA_ERRCHK(cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_y, y, sizeof(double) * N, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice));

  // execute functions and measure time
  //do it four times as I measure combined exec time of 4 functions
  //for convenience do not re-initialize cuda_alpha oor use a nex one since this is just for benchmarking
  CUDA_ERRCHK(cudaDeviceSynchronize()); 
  timer.reset(); 
    cuda_dot_product<<<512, 512>>>(N, cuda_x, cuda_y, cuda_alpha);
    cuda_dot_product<<<512, 512>>>(N, cuda_x, cuda_y, cuda_alpha);
    cuda_dot_product<<<512, 512>>>(N, cuda_x, cuda_y, cuda_alpha);
    cuda_dot_product<<<512, 512>>>(N, cuda_x, cuda_y, cuda_alpha);
  CUDA_ERRCHK(cudaDeviceSynchronize());  double elapsed = timer.get(); 

  std:: cout << N << "," << elapsed << std::endl;

  CUDA_ERRCHK(cudaMemcpy(&alpha, cuda_alpha, sizeof(double), cudaMemcpyDeviceToHost));

  // Clean up
  CUDA_ERRCHK(cudaFree(cuda_x));
  CUDA_ERRCHK(cudaFree(cuda_y));
  CUDA_ERRCHK(cudaFree(cuda_alpha));
  free(x);
  free(y);
  }
  return EXIT_SUCCESS;
}

