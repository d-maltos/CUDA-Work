# CUDA Experiments
Author: Dylan Maltos
- Just some cool things I've done via experimenting with CUDA

## sobel_filter
A sobel image filter that utilizes GPU resource via CUDA programming
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
