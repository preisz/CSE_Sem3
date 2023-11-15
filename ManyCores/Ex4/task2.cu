#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdio.h>

#define BLOCKSIZE 8 // kernel with 8 threads per block 
//use shared memory for efficient data sharing among threads within a block

__global__ void cuda_dot_product(int N, double *x, double *y, double *result) {
    __shared__ double shared_mem[BLOCKSIZE];// Shared memory to store partial dot products (8)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double partialSum = 0.0;// Initialize partial sum to zero

    // Compute partial dot product for each thread
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {partialSum += x[i] * y[i];}
    
    shared_mem[threadIdx.x] = partialSum;     // Store partial sum in shared memory
    __syncthreads();

    // Perform reduction to calculate the final dot product
    if (threadIdx.x == 0) {
        double blockSum = 0.0;
        for (int i = 0; i < blockDim.x; ++i) {
            blockSum += shared_mem[i];
        }
        atomicAdd(result, blockSum);
    }
}

int main() {

  Timer timer;
  std::vector<int> Nvals = { 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000, 50'000'000, 100'000'000, };
  std::cout << "****Dot Product****\n";
  std::cout << "Length of vector N, Execution time for operations" << std::endl;

  for (int N : Nvals){
  // Allocate and initialize arrays on CPU

  int K = 16; // K is a multiple of 8
  int numBlocks = (N + BLOCKSIZE - 1) / BLOCK_SIZE;

   for (int i = 0; i < K; i += BLOCKSIZE) { // Launch the kernel for K/8 iterations
        cuda_dot_product<<<numBlocks, BLOCKSIZE>>>(x, y + k * vectorSize, dev_C + k, vectorSize);
    }

  double *x = (double *)malloc(sizeof(double) * N);
  //double *y = (double *)malloc(sizeof(double) * N);
   double alpha = 0;

  std::fill(x, x + N, 1);
  
  std::fill(y, y + N, 1);

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

  int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;

  // execute functions and measure time
  CUDA_ERRCHK(cudaDeviceSynchronize()); 
  timer.reset(); 
    cuda_dot_product<<<numBlocks, BLOCKSIZE>>>(N, cuda_x, cuda_y, cuda_alpha);
 CUDA_ERRCHK(cudaDeviceSynchronize());  double elapsed = timer.get(); 

  std:: cout << N << "," << elapsed << std::endl;

  CUDA_ERRCHK(cudaMemcpy(&alpha, cuda_alpha, sizeof(double), cudaMemcpyDeviceToHost));
std:: cout << "Result: " << alpha << std::endl;


  // Clean up
  CUDA_ERRCHK(cudaFree(cuda_x));
  CUDA_ERRCHK(cudaFree(cuda_y));
  CUDA_ERRCHK(cudaFree(cuda_alpha));
  free(x);
  free(y);
  }
  return EXIT_SUCCESS;
}
