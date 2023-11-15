#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cmath> //abs

#define BLOCKSIZE 256

__global__ void initAll( double* cuda_alpha, double* cuda_oneNorm, double* cuda_twoNorm, unsigned* cudaNumZeros  ){
    *cuda_alpha = 0.0;
    *cuda_oneNorm = 0.0;
    *cuda_twoNorm = 0.0;
    *cudaNumZeros = 0;
}

__global__ void sumVectorKernel(const double* x, int N,  double* result) {    //calculate the sum of entries
    __shared__ double shared_mem[BLOCKSIZE];

    int localtid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

   double sum = 0;
   for (int i = gid; i < N; i += blockDim.x * gridDim.x) {sum += x[i]; }

  shared_mem[localtid] = sum;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[localtid] += shared_mem[localtid + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}


__global__ void OneNormKernel(const double* x, int N, double* result) {   //calculate one norm
    __shared__ double shared_mem[BLOCKSIZE];

    int localtid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

   double sum = 0;
   for (int i = gid; i < N; i += blockDim.x * gridDim.x) {sum += std::abs(x[i]); }

  shared_mem[localtid] = sum;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[localtid] += shared_mem[localtid + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}

__global__ void TwoNormKernel(const double* x, int N, double* result) {  //calculate two  norm
    __shared__ double shared_mem[BLOCKSIZE];

    int localtid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

   double sum = 0;
   for (int i = gid; i < N; i += blockDim.x * gridDim.x) {sum += x[i] * x[i] ; }

  shared_mem[localtid] = sum;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[localtid] += shared_mem[localtid + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}


__global__ void ZeroCounterKernel(const double* x, int N, unsigned* result, double tol) {  //calculate 0 entries ==> smaller than tol
    __shared__ double shared_mem[BLOCKSIZE];

    int localtid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

   unsigned numZeros = 0;
   for (int i = gid; i < N; i += blockDim.x * gridDim.x) {
        if( std::abs(x[i])<tol ) 
            numZeros ++ ; 
    }

  shared_mem[localtid] = numZeros;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[localtid] += shared_mem[localtid + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}

int main(void){
  //int N = 1'000'000;
  Timer timer;
  std::vector<int> Nvals = { 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000, 50'000'000, 100'000'000, };
  std::cout << "****Using shared memory****\n";
  std::cout << "Length of vector N, Execution time for operations" << std::endl;

  for (int N : Nvals)   {
    // Allocate and initialize arrays on CPU
  double *x = (double *)malloc(sizeof(double) * N);
  double alpha = 0.0, oneNorm = 0.0, twoNorm = 0.0;
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
  double elapsed = timer.get(); // wait for kernel to finish, then print elapsed time

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
  CUDA_ERRCHK(cudaFree(cuda_alpha));CUDA_ERRCHK(cudaFree(cuda_OneNorm)); CUDA_ERRCHK(cudaFree(cuda_TwoNorm)); 
  CUDA_ERRCHK(cudaFree(cuda_numzeros));

  free(x);

  }
  return EXIT_SUCCESS;

}