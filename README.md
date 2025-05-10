# CUDA Experiments
Author: Dylan Maltos
- Just some cool things I've done via experimenting with CUDA

## sobel_filter
A sobel image filter that utilizes GPU resource via CUDA programming
- Uses a tiled 2D array of blocks and a 2D array of threads per block
- Works for images of all different sizes

Compile and run with:
```nvcc sobel-gpu.cu -lfreeimage```