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

  double *x = (double *)malloc(sizeof(double) * N);
  std::fill(x, x + N, 1);

  //double** y = new double*[K];// Allocate a single dynamic array for the rows
        //for (int i = 0; i < K; ++i) { y[i] = new double[N];  } //y[k] is a vector of row N

    // Create a vector of double arrays. Push arrays later
    std::vector<double*> y;
    for (int k = 0; k < K; ++k) {
        double* yk = new double[N];
        std::fill(yk, yk + N, k); //Init
        y.push_back(yk);
    }
   std::vector<double> alpha(K, 0.0); //K results


  // Allocate and initialize arrays on GPU
  double *cuda_x;
  double *cuda_y;
  double *cuda_alpha;

  CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N));
  CUDA_ERRCHK(cudaMalloc(&cuda_alpha, sizeof(double) * K ));
  CUDA_ERRCHK(cudaMalloc(&cuda_y, N * K * sizeof(double)));

   for (int k = 0; k < K; ++k) {
        cudaMemcpy(&cuda_y[k * N], y[k], N * sizeof(double), cudaMemcpyHostToDevice);
    }
  CUDA_ERRCHK(cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_alpha, &alpha, sizeof(double) * K, cudaMemcpyHostToDevice));

  // execute functions and measure time
  CUDA_ERRCHK(cudaDeviceSynchronize()); 
  int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
  timer.reset(); int i =0;
  
    for (int k = 0; k < K; k += BLOCKSIZE) {  // Launch the kernel for K/8 iterations
        cuda_dot_product<<<numBlocks, BLOCKSIZE>>>(N, cuda_x, cuda_y[k], cuda_alpha + k );
    }
     



  /*int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
        for (int k = 0; k < K; k += BLOCKSIZE) {  // Launch the kernel for K/8 iterations
            cuda_dot_product<<<numBlocks, BLOCKSIZE>>>(N, cuda_x, cuda_y[k], cuda_alpha + k );
        }*/
   CUDA_ERRCHK(cudaDeviceSynchronize());  double elapsed = timer.get(); 

  std:: cout << N << "," << elapsed << std::endl;

  CUDA_ERRCHK(cudaMemcpy(&alpha, cuda_alpha, sizeof(double) * K, cudaMemcpyDeviceToHost));
  std:: cout << "\nResults:\n " ;
  for (const auto& res : alpha) {std::cout << res << ", ";}

  // Clean up
  CUDA_ERRCHK(cudaFree(cuda_x));
  CUDA_ERRCHK(cudaFree(cuda_y));
  CUDA_ERRCHK(cudaFree(cuda_alpha));
  free(x);
  for(size_t k=0, k<K, k++)
    delete[] y[k];
  }
  return EXIT_SUCCESS;
}
