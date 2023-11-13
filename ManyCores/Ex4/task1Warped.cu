#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <cmath> //abs

#define BLOCKSIZE 256

/*__global__ void sumVectorKernel(const double* x, int N, double* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = threadIdx.x / warpSize;
    std::cout << "Warpsize: " << warpSize << std::endl;

    double sum = x[tid];

    // Warp shuffle reduction
    for (int i = warpSize; i > 0; i /= 2) {
        sum += __shfl_down_sync(-1, sum, i);
    }
     if (threadIdx.x % warpSize == 0) {atomicAdd(result, sum);}// thread 0 contains sum of all values within the warp
}*/


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

__global__ void OneNormKernel(const double* x, int N, double* result) {   //calculate one norm
    int tid = threadIdx.x;
    int warpIdx = threadIdx.x / warpSize;

   double sum = 0;
   for (int i = tid; i < N; i += blockDim.x * gridDim.x) {sum += std::abs(x[i]);}

    // Warp shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // The first thread in each warp adds its partial sum to the global result
    if (threadIdx.x % warpSize == 0) {atomicAdd(result, sum);}
}


__global__ void TwoNormKernel(const double* x, int N, double* result) {   //calculate 2- norm
    int tid = threadIdx.x;
    int warpIdx = threadIdx.x / warpSize;

   double sum = 0;
   for (int i = tid; i < N; i += blockDim.x * gridDim.x) {sum += x[i] * x[i] ;}

    // Warp shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // The first thread in each warp adds its partial sum to the global result
    if (threadIdx.x % warpSize == 0) {atomicAdd(result, sum);}
}

__global__ void ZeroCounterKernel(const double* x, int N, unsigned* result, double tol) {   //calc 0 entries==> smaller than tol
    int tid = threadIdx.x;
    int warpIdx = threadIdx.x / warpSize;

   unsigned numzeros = 0;
   for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        if( std::abs(x[i]) < tol )
            numzeros ++;
    }

    // Warp shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        numzeros += __shfl_down_sync(0xFFFFFFFF, numzeros, offset);
    }

    // The first thread in each warp adds its partial sum to the global result
    if (threadIdx.x % warpSize == 0) {atomicAdd(result, numzeros);}
}

int main(void){
  int N = 1'000'000;
     
    // Allocate and initialize arrays on CPU
  double *x = (double *)malloc(sizeof(double) * N);
  double alpha, oneNorm, twoNorm = 0;
  unsigned numzeros = 0;

  std::fill(x, x + 5000, 1);
  //std::fill( x + 5000, x + 5000 + 5000, -1);


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

  
  sumVectorKernel<<<BLOCKSIZE, BLOCKSIZE>>>(cuda_x, N, cuda_alpha);
  OneNormKernel<<<BLOCKSIZE, BLOCKSIZE>>>(cuda_x, N, cuda_OneNorm);
  TwoNormKernel<<<BLOCKSIZE, BLOCKSIZE>>>(cuda_x, N, cuda_TwoNorm);
  ZeroCounterKernel<<<BLOCKSIZE, BLOCKSIZE>>>(cuda_x, N, cuda_numzeros, 1e-9);

  
  CUDA_ERRCHK(cudaMemcpy(&alpha, cuda_alpha, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_ERRCHK(cudaMemcpy(&oneNorm, cuda_OneNorm, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_ERRCHK(cudaMemcpy(&twoNorm, cuda_TwoNorm, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_ERRCHK(cudaMemcpy(&numzeros, cuda_numzeros, sizeof(unsigned), cudaMemcpyDeviceToHost));

  std::cout << "Result of summing entries: " << alpha << std::endl;
  std::cout << "Result of 1-NORM: " << oneNorm << std::endl;
  std::cout << "Result of 2-NORM: " << twoNorm << std::endl;
  std::cout << "Result of counting zero entries: " << numzeros << std::endl;

  // Clean up
  CUDA_ERRCHK(cudaFree(cuda_x));
  CUDA_ERRCHK(cudaFree(cuda_alpha));CUDA_ERRCHK(cudaFree(cuda_OneNorm)); CUDA_ERRCHK(cudaFree(cuda_TwoNorm)); CUDA_ERRCHK(cudaFree(cuda_numzeros));

  free(x);

  return EXIT_SUCCESS;


}
