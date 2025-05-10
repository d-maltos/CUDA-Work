/***********************************************************************
 * raytracer-gpu.cu
 *
 * Dylan Maltos
 * 04-06-2025
 *
 * Program description:
 * - A GPU-accelerated ray tracer that uses CUDA for parallel rendering
 * - Casts rays top-down across a 2048x2048 image grid
 * - Each ray tests intersection with 80 randomly generated spheres
 * - Pixel color is determined by sphere hit location and normal
 * - Saves the rendered output as 'rayGPU.png' using FreeImage
 * - Measures and reports GPU render time for performance comparison
 *
 * Compile/run with:  nvcc raytracer-gpu.cu -lfreeimage -o raytracer-gpu && ./raytracer-gpu
 ***********************************************************************/

// Header files
#include "FreeImage.h"
#include <stdio.h>

// Defines
#define DIM 2048
#define SPHERES 80
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;

    // CUDA device function for ray hit check
    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / radius;
            return dz + z;
        }
        return -INF;
    }
};

// Put the spheres in fast constant memory
__constant__ Sphere d_spheres[SPHERES]; 

// CUDA kernel to render each pixel
__global__ void draw(char *red, char *green, char *blue) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DIM || y >= DIM) return;

    int offset = x + y * DIM;
    float ox = x - DIM / 2;
    float oy = y - DIM / 2;

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; ++i) {
        float n;
        float t = d_spheres[i].hit(ox, oy, &n);
        if (t > maxz) {
            r = d_spheres[i].r * n;
            g = d_spheres[i].g * n;
            b = d_spheres[i].b * n;
            maxz = t;
        }
    }

    red[offset] = (char)(r * 255);
    green[offset] = (char)(g * 255);
    blue[offset] = (char)(b * 255);
}

void saveImage(char *red, char *green, char *blue) {
    FreeImage_Initialise();
    FIBITMAP *bitmap = FreeImage_Allocate(DIM, DIM, 24);

    RGBQUAD color;
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int index = y * DIM + x;
            color.rgbRed = red[index];
            color.rgbGreen = green[index];
            color.rgbBlue = blue[index];
            FreeImage_SetPixelColor(bitmap, x, y, &color);
        }
    }

    FreeImage_Save(FIF_PNG, bitmap, "rayGPU.png", 0);
    FreeImage_Unload(bitmap);
    FreeImage_DeInitialise();
}

int main() {
    srand(time(NULL));

    // Host side sphere creation
    Sphere h_spheres[SPHERES];
    for (int i = 0; i < SPHERES; ++i) {
        h_spheres[i].r = rnd(1.0f);
        h_spheres[i].g = rnd(1.0f);
        h_spheres[i].b = rnd(1.0f);
        h_spheres[i].x = rnd((float)DIM) - DIM / 2;
        h_spheres[i].y = rnd((float)DIM) - DIM / 2;
        h_spheres[i].z = rnd((float)DIM) - DIM / 2;
        h_spheres[i].radius = rnd(200.0f) + 40;
    }

    // Copy spheres to constant memory
    cudaMemcpyToSymbol(d_spheres, h_spheres, sizeof(Sphere) * SPHERES);

    // Allocate memory
    char *d_red, *d_green, *d_blue;
    cudaMalloc(&d_red, DIM * DIM);
    cudaMalloc(&d_green, DIM * DIM);
    cudaMalloc(&d_blue, DIM * DIM);

    // Time the GPU render
    clock_t start = clock();

    // Launch kernel: 2D grid and block
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(DIM / 16, DIM / 16);
    draw<<<blocks, threadsPerBlock>>>(d_red, d_green, d_blue);
    cudaDeviceSynchronize();

    clock_t end = clock();
    double gpuTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Copy back to host using memcpy
    char *h_red = (char*)malloc(DIM * DIM);
    char *h_green = (char*)malloc(DIM * DIM);
    char *h_blue = (char*)malloc(DIM * DIM);
    cudaMemcpy(h_red, d_red, DIM * DIM, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_green, d_green, DIM * DIM, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blue, d_blue, DIM * DIM, cudaMemcpyDeviceToHost);

    // Save the image as rayGPU.png
    saveImage(h_red, h_green, h_blue);
    printf("GPU Time: %.3f sec\n", gpuTime);

    // Cleanup and free the memory
    free(h_red);
    free(h_green);
    free(h_blue);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);

    // Gracefully exit the program
    return 0;
}