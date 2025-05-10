/***********************************************************************
 * parallel-minimum-mpi.cpp
 *
 * Dylan Maltos
 * 04-06-2025
 *
 * Program description:
 * - Finds the global minimum of 8 million random integers using MPI
 * - Splits the dataset across 8 processes with MPI_Scatter
 * - Each process computes a local minimum
 * - Process 0 gathers local mins via MPI_Reduce and outputs the global min
 * - Validates result by computing the sequential minimum
 *
 * Compile/run with:  mpic++ -o parallel-minimum parallel-minimum-mpi.cpp && mpirun -np 8 ./parallel-minimum
 ***********************************************************************/

// Header files
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>   
#include <cstdlib>

// Global Variables: SIZE - Total number of elements; MAX_VALUE - Maxmimum random integer value
const int SIZE = 8000000;
const int MAX_VALUE = 1000000000; 

// Initialize the MPI environment
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); 

    int process_rank, process_size;
    // Get current process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    // Get total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);  

    std::vector<int> data;

    // Process 0 is the only process that generates the data
    if (process_rank == 0) {
        srand(time(NULL));
        data.resize(SIZE);
        for (int i = 0; i < SIZE; ++i)
            data[i] = rand() % MAX_VALUE;
    }

    // Divide the data among the 8 processes
    int chunk_size = SIZE / process_size;  
    std::vector<int> local_data(chunk_size);

    // Scatter the data to all processes (each gets chunk_size elements)
    MPI_Scatter(data.data(), chunk_size, MPI_INT,
                local_data.data(), chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    // Each process finds the minimum in its local chunk
    int local_min = *std::min_element(local_data.begin(), local_data.end());

    // Reduce local minimums to find the global minimum in process 0
    int global_min;
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (process_rank == 0) {
        // Validate by finding minimum sequentially.
        int seq_min = *std::min_element(data.begin(), data.end());
        std::cout << "Parallel Min: " << global_min << "\n";
        std::cout << "Sequential Min: " << seq_min << "\n";
        std::cout << "Match: " << (global_min == seq_min ? "Yes" : "No") << "\n";
    }

    // Cleanup
    MPI_Finalize();  
    return 0;
}