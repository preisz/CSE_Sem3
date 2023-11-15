#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <cmath> //abs

#define BLOCKSIZE 256


__global__ void sumVectorKernel(const double* x, int N, double* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % warpSize;
    //std::cout << "Warpsize: " << warpSize << std::endl;

    double sum = x[tid];

    // Warp shuffle reduction
    for (int i = warpSize / 2; i > 0; i /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, i, warpSize);
    }

    // The first thread in each warp adds its partial sum to the global result
    if (laneId == 0) {
        atomicAdd(result, sum);
    }
}

__global__ void OneNormKernel(const double* x, int N, double* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % warpSize;

    double sum = std::abs(x[tid]);

    // Warp shuffle reduction
    for (int i = warpSize / 2; i > 0; i /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, i, warpSize);
    }

    // The first thread in each warp adds its partial sum to the global result
    if (laneId == 0) {
        atomicAdd(result, sum);
    }
}

__global__ void TwoNormKernel(const double* x, int N, double* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % warpSize;

    double sum = x[tid] * x[tid];

    // Warp shuffle reduction
    for (int i = warpSize / 2; i > 0; i /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, i, warpSize);
    }

    // The first thread in each warp adds its partial sum to the global result
    if (laneId == 0) {
        atomicAdd(result, sum);
    }
}

__global__ void ZeroCounterKernel(const double* x, int N, unsigned* result, double tol) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % warpSize;
    unsigned numzeros = 0;

    if (std::abs(x[tid]) < tol && (tid < N) ) {numzeros = 1;}

    // Warp shuffle reduction
    for (int i = warpSize / 2; i > 0; i /= 2) {
        numzeros += __shfl_down_sync(0xFFFFFFFF, numzeros, i, warpSize);
    }

    // The first thread in each warp adds its partial sum to the global result
    if (laneId == 0) 
        atomicAdd(result, numzeros);
}


int main(void){
  Timer timer;
  std::vector<int> Nvals = { 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000, 50'000'000, 100'000'000, };
  std::cout << "****Using warps****\n";
  std::cout << "Length of vector N, Execution time for operations" << std::endl;

  for (int N : Nvals){
     
    // Allocate and initialize arrays on CPU
  double *x = (double *)malloc(sizeof(double) * N);
  double alpha, oneNorm, twoNorm = 0;
  unsigned numzeros = 0;

  int a = int(N/4);
  std::fill(x, x + a, 2);
  std::fill( x + a, x + (a + a + a), -1);

  // Allocate and initialize arrays on GPU
  double *cuda_x;
  double *cuda_alpha, *cuda_OneNorm, *cuda_TwoNorm;
  unsigned *cuda_numzeros;
  
  CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N));
  CUDA_ERRCHK(cudaMalloc(&cuda_alpha, sizeof(double)));
  CUDA_ERRCHK(cudaMalloc(&cuda_OneNorm, sizeof(double)));
  CUDA_ERRCHK(cudaMalloc(&cuda_TwoNorm, sizeof(double)));
  CUDA_ERRCHK(cudaMalloc(&cuda_numzeros, sizeof(unsigned)));

  
  CUDA_ERRCHK(cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_OneNorm, &oneNorm, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_TwoNorm, &twoNorm, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_numzeros, &numzeros, sizeof(unsigned), cudaMemcpyHostToDevice));

  // execute functions and measure time
  CUDA_ERRCHK(cudaDeviceSynchronize());   
  timer.reset(); 
    sumVectorKernel<<<BLOCKSIZE, BLOCKSIZE>>>(cuda_x, N, cuda_alpha);
    OneNormKernel<<<BLOCKSIZE, BLOCKSIZE>>>(cuda_x, N, cuda_OneNorm);
    TwoNormKernel<<<BLOCKSIZE, BLOCKSIZE>>>(cuda_x, N, cuda_TwoNorm);
    ZeroCounterKernel<<<BLOCKSIZE, BLOCKSIZE>>>(cuda_x, N, cuda_numzeros, 1e-9);
  CUDA_ERRCHK(cudaDeviceSynchronize());   
  double elapsed = timer.get();

  std:: cout << N << "," << elapsed << std::endl;
 
  CUDA_ERRCHK(cudaMemcpy(&alpha, cuda_alpha, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_ERRCHK(cudaMemcpy(&oneNorm, cuda_OneNorm, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_ERRCHK(cudaMemcpy(&twoNorm, cuda_TwoNorm, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_ERRCHK(cudaMemcpy(&numzeros, cuda_numzeros, sizeof(unsigned), cudaMemcpyDeviceToHost));

/* DEBUG PART
  std::cout << "Result of summing entries: " << alpha << std::endl;
  std::cout << "Result of 1-NORM: " << oneNorm << std::endl;
  std::cout << "Result of 2-NORM: " << twoNorm << std::endl;
  std::cout << "Result of counting zero entries: " << numzeros << std::endl;
*/

  // Clean up
  CUDA_ERRCHK(cudaFree(cuda_x));
  CUDA_ERRCHK(cudaFree(cuda_alpha));CUDA_ERRCHK(cudaFree(cuda_OneNorm)); CUDA_ERRCHK(cudaFree(cuda_TwoNorm)); CUDA_ERRCHK(cudaFree(cuda_numzeros));

  free(x);
  }
  return EXIT_SUCCESS;


}
