# Report Exercise 4

## Task 1

The version a) when using shared memory looks like this
```c++
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

```

The version of b) when using warps looks like this

```c++
__global__ void sumVectorKernel(const double* x, int N, double* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % warpSize;

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

```

When we want to use one thread per entry then we use the same function as in b) just adjust the grid size:

```c++
  dim3 blockSize(32);  // One thread per entry
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x);  // Adjust grid size based on vector size
   
   //---- more code 
  sumVectorKernel<<<gridSize, blockSize>>>(cuda_x, N, cuda_alpha);
```

The results can be seen in the figure below:.
![Alt text](image.png)

# Task 2

Due to lack of time I did not manage to finish this task. The codes can be found in *task2.cu*.