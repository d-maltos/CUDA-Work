# CUDA and MPI Experiments
Author: Dylan Maltos
- Just some cool things I've done via experimenting with CUDA and MPI

## sobel-filter
Sobel image filter that utilizes GPU resources via CUDA
- Uses a tiled 2D array of blocks and a 2D array of threads per block
- Works for images of all different sizes

Compile and run with:
```
nvcc sobel-gpu.cu -lfreeimage
```

## parallel-minimum-mpi
Finds the global minimum of 8 million random integers using MPI
- Splits the dataset across 8 processes
- Each process computes a local minimum
- Process 0 gathers local minimums via MPI_Reduce and outputs the global minimum
- Valdidates result by computer the sequential minimum

Compile and run with:
```
mpic++ -o parallel-minimum parallel-minimum-mpi.cpp && mpirun -np 8 ./parallel-minimum
```

## parallel-minimum-cuda
CUDA program to find the global minimum of 8 million random integers
- Launches 8 GPU threads, each computing alocal minimum over 1/8th of the data
- Final reduction of partial minimums is done on the host
- Validates result by computing the minimum sequentially on the CPU

Compile and run with:
```
 nvcc parallel-minimum-cuda.cu -o parallel-minimum-cuda && ./parallel-minimum-cuda
```

## 2d-array-sum
CUDA program to compute the total sum of a 3x4 matrix
- Each GPU thread sums one column of the matrix in parallel
- The host aggregates the column sums to get the final total
- Outputs the matrix, column-wise sums, and overall total

Compile and run with:
```
nvcc 2d-array-sum.cu -o 2d-array-sum && ./2d-array-sum
```

## raytracer-gpu
GPU-accelerated ray tracer that uses CUDA for parallel rendering
- Casts rays top-down across a 2048x2048 image grid
- Each ray tests intersection with 80 randomly generated spheres
- Pixel color is determined by sphere hit location and normal
- Saves the rendered output as 'rayGPU.png' using FreeImage
- Measures and reports GPU render time for performance comparison

Compile and run with:
```
nvcc raytracer-gpu.cu -lfreeimage -o raytracer-gpu && ./raytracer-gpu
```