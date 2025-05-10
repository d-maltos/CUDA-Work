/***********************************************************************
 * 2d-array-sum.cu
 *
 * Dylan Maltos
 * 04-06-2025
 *
 * Program description:
 * - A CUDA program that computes the total sum of a 3x4 matrix
 * - Each GPU thread sums one column of the matrix in parallel
 * - The host aggregates the column sums to get the final total
 * - Outputs the matrix, column-wise sums, and overall total
 *
 * Compile/run with:  nvcc 2d-array-sum.cu -o 2d-array-sum && ./2d-array-sum
 ***********************************************************************/

// Header files
#include <iostream>

// Defines: ROWS - 3 rows; COLS - 4 columns
#define ROWS 3
#define COLS 4

// Kernel function to get the sum of each column in the 3x4 matrix
__global__ void columnSum(int *matrix, int *colSums) {
    // Each thread handles one column of the 3x4 matrix
    int col = threadIdx.x; 
    if (col < COLS) {
        int sum = 0;
        // Sum over all rows in the current column
        for (int row = 0; row < ROWS; ++row) {
            // Put them in row-major order
            sum += matrix[row * COLS + col]; 
        }
        // Store the column sum
        colSums[col] = sum; 
    }
}

int main() {
    // Define the 3x4 host matrix
    int h_matrix[ROWS][COLS] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    int *d_matrix, *d_colSums;
    int h_colSums[COLS];

    // Allocate memory
    cudaMalloc(&d_matrix, ROWS * COLS * sizeof(int));
    cudaMalloc(&d_colSums, COLS * sizeof(int));

    // Copy the matrix from host to device using memcpy
    cudaMemcpy(d_matrix, h_matrix, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: 1 block, 1 thread per column (4 total)
    columnSum<<<1, COLS>>>(d_matrix, d_colSums);

    // Copy column sums back to the host
    cudaMemcpy(h_colSums, d_colSums, COLS * sizeof(int), cudaMemcpyDeviceToHost);

    // Add up all the column sums on the host to get the total sum
    int total = 0;
    
    // Include printing the matrix in the program output
    std::cout << "Matrix:\n";
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            std::cout << h_matrix[i][j] << "\t";
        }
        std::cout << "\n";
    }

    // Include printing out the sum of each column in the matrix in the program output
    std::cout << "Column sums:\n";
    for (int i = 0; i < COLS; ++i) {
        std::cout << "Column " << i << ": " << h_colSums[i] << std::endl;
        total += h_colSums[i];
    }

    // Include printing the total sum in the program output
    std::cout << "Total sum: " << total << std::endl;

    // Cleanup and free the memory
    cudaFree(d_matrix);
    cudaFree(d_colSums);

    // Gracefully exit the program
    return 0;
}