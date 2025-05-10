/***********************************************************************
 * parallel-minimum-cuda.cu
 *
 * Dylan Maltos
 * 04-06-2025
 *
 * Program description:
 * - A CUDA program to find the global minimum of 8 million random integers
 * - Launches 8 GPU threads, each computing a local minimum over 1/8th of the data
 * - Final reduction of partial minimums is done on the host
 * - Validates result by computing the minimum sequentially on the CPU
 *
 * Compile/run with:  nvcc parallel-minimum-cuda.cu -o parallel-minimum-cuda && ./parallel-minimum-cuda
 ***********************************************************************/

// Header files
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define SIZE 8000000
#define THREADS 8
#define MAX_VALUE 1000000000
#define CHUNK_SIZE (SIZE / THREADS)

// CUDA kernel to compute local minimum per thread
__global__ void min(int *nums, int *partialMin) {
    int start_idx = threadIdx.x * CHUNK_SIZE;
    int end_idx   = start_idx + CHUNK_SIZE;

    int minSoFar = nums[start_idx];
    for (int i = start_idx; i < end_idx; ++i) {
        if (nums[i] < minSoFar)
            minSoFar = nums[i];
    }

    partialMin[threadIdx.x] = minSoFar;
}

int main() {
    srand(time(NULL));

    // Allocate and fill the host array
    int *nums = (int*) malloc(sizeof(int) * SIZE);
    for (int i = 0; i < SIZE; ++i) {
        nums[i] = rand() % MAX_VALUE;
    }

    // Serial CPU-based minimum
    int serialMin = nums[0];
    for (int i = 1; i < SIZE; ++i) {
        if (nums[i] < serialMin)
            serialMin = nums[i];
    }
    printf("Serial min: %d\n", serialMin);

    // Allocate the device memory
    int *dev_nums, *dev_partialMin;
    cudaMalloc((void**)&dev_nums, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_partialMin, THREADS * sizeof(int));

    // Allocate the host memory for partial results
    int *partialMin = (int*) malloc(sizeof(int) * THREADS);

    // Copy data from host to device using memcpy
    cudaMemcpy(dev_nums, nums, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: 1 block, 8 threads
    min<<<1, THREADS>>>(dev_nums, dev_partialMin);

    // Copy the results back
    cudaMemcpy(partialMin, dev_partialMin, THREADS * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on host
    int parallelMin = partialMin[0];
    for (int i = 1; i < THREADS; ++i) {
        if (partialMin[i] < parallelMin)
            parallelMin = partialMin[i];
    }

    printf("Parallel min: %d\n", parallelMin);
    printf("Match: %s\n", (parallelMin == serialMin ? "Yes" : "No"));

    // Cleanup and free the memory
    cudaFree(dev_nums);
    cudaFree(dev_partialMin);
    free(nums);
    free(partialMin);

    // Gracefully exit the program
    return 0;
}