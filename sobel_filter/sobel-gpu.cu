/***********************************************************************
 * sobel-gpu.cu
 *
 * Dylan Maltos
 * 04-26-2025
 *
 * Program description:
 * - A sobel image filter using GPU via CUDA
 * - Uses a tiled 2D array of blocks and a 2D array of threads per block
 * - Solution works even for different sized images
 *
 * Compile/run with:  nvcc sobel-gpu.cu -lfreeimage
 ***********************************************************************/

 #include "FreeImage.h"
 #include <stdio.h>
 #include <math.h>
 
 #define TILE_SIZE 16
 
 // Returns the index into the 1d pixel array
 __device__ __host__ int pixelIndex(int x, int y, int width)
 {
  return (y*width + x);
 }
 
 // Returns the sobel value for pixel x,y
 __device__ int sobel(int x, int y, int width, unsigned char *pixels)
 {
 int x00 = -1;  int x20 = 1;
 int x01 = -2;  int x21 = 2;
 int x02 = -1;  int x22 = 1;
 x00 *= pixels[pixelIndex(x-1,y-1,width)];
 x01 *= pixels[pixelIndex(x-1,y,width)];
 x02 *= pixels[pixelIndex(x-1,y+1,width)];
 x20 *= pixels[pixelIndex(x+1,y-1,width)];
 x21 *= pixels[pixelIndex(x+1,y,width)];
 x22 *= pixels[pixelIndex(x+1,y+1,width)];
 
 int y00 = -1;  int y10 = -2;  int y20 = -1;
 int y02 = 1;   int y12 = 2;   int y22 = 1;
 y00 *= pixels[pixelIndex(x-1,y-1,width)];
 y10 *= pixels[pixelIndex(x,y-1,width)];
 y20 *= pixels[pixelIndex(x+1,y-1,width)];
 y02 *= pixels[pixelIndex(x-1,y+1,width)];
 y12 *= pixels[pixelIndex(x,y+1,width)];
 y22 *= pixels[pixelIndex(x+1,y+1,width)];
 
 int px = x00 + x01 + x02 + x20 + x21 + x22;
 int py = y00 + y10 + y20 + y02 + y12 + y22;
 return sqrtf((float)(px*px + py*py));
 }
 
 // Kernel applies sobel filter
 __global__ void sobelKernel(unsigned char *input, unsigned char *output, int width, int height)
 {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
 
  if (x > 0 && y > 0 && x < width-1 && y < height-1)
  {
    int sVal = sobel(x,y,width,input);
    if (sVal > 255) sVal = 255;
    if (sVal < 0) sVal = 0;
    output[pixelIndex(x,y,width)] = (unsigned char)sVal;
  }
 }
 
 int main()
 {
  FreeImage_Initialise();
  atexit(FreeImage_DeInitialise);
 
  // Load image and get width and height
  FIBITMAP *image;
  // Replace "source_image.png" with the image you want to apply the sobel filter too
  image = FreeImage_Load(FIF_PNG, "source_image.png", 0);
  if (image == NULL)
  {
    printf("Image Load Problem\n");
    exit(0);
  }
  int imgWidth;
  int imgHeight;
  imgWidth = FreeImage_GetWidth(image);
  imgHeight = FreeImage_GetHeight(image);
 
  // Convert image into flat array
  RGBQUAD aPixel;
  unsigned char *pixels;
  int pixIndex = 0;
  pixels = (unsigned char *) malloc(sizeof(unsigned char)*imgWidth*imgHeight);
  for (int i = 0; i < imgHeight; i++)
   for (int j = 0; j < imgWidth; j++)
   {
     FreeImage_GetPixelColor(image,j,i,&aPixel);
     unsigned char grey = ((aPixel.rgbRed + aPixel.rgbGreen + aPixel.rgbBlue)/3);
     pixels[pixIndex++]=grey;
   }
 
  // Allocate device memory
  unsigned char *d_pixels, *d_output;
  cudaMalloc((void**)&d_pixels, sizeof(unsigned char)*imgWidth*imgHeight);
  cudaMalloc((void**)&d_output, sizeof(unsigned char)*imgWidth*imgHeight);
  cudaMemcpy(d_pixels, pixels, sizeof(unsigned char)*imgWidth*imgHeight, cudaMemcpyHostToDevice);
 
  // Launch kernel
  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize((imgWidth + TILE_SIZE - 1)/TILE_SIZE, (imgHeight + TILE_SIZE - 1)/TILE_SIZE);
  sobelKernel<<<gridSize, blockSize>>>(d_pixels, d_output, imgWidth, imgHeight);
  cudaDeviceSynchronize();
 
  cudaMemcpy(pixels, d_output, sizeof(unsigned char)*imgWidth*imgHeight, cudaMemcpyDeviceToHost);
 
  // Save output
  FIBITMAP *bitmap = FreeImage_Allocate(imgWidth, imgHeight, 24);
  pixIndex = 0;
  for (int i = 0; i < imgHeight; i++)
   for (int j = 0; j < imgWidth; j++)
   {
     int sVal = pixels[pixelIndex(j, i, imgWidth)];
     aPixel.rgbRed = sVal;
     aPixel.rgbGreen = sVal;
     aPixel.rgbBlue = sVal;
     FreeImage_SetPixelColor(bitmap, j, i, &aPixel);
   }

  // Replace "filtered_image.png" with whatever you want the name of the finished filtered image to be
  FreeImage_Save(FIF_PNG, bitmap, "filtered_image.png", 0);
 
  free(pixels);
  cudaFree(d_pixels);
  cudaFree(d_output);
  FreeImage_Unload(bitmap);
  FreeImage_Unload(image);
  return 0;
 }