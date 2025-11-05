#include "BRKGA_GPU.hpp"
#include "BinPacking3D_cuda.hpp"
#include "cuda_structures.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include <random>
#include <chrono>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>

__device__ double* d_population_global = nullptr;
__device__ double* d_fitness_list_global = nullptr;
__device__ int* d_elites_global = nullptr;
__device__ int* d_non_elites_global = nullptr;
__device__ int d_num_individuals_global = 0;
__device__ int d_num_gene_global = 0;
__device__ int d_num_elites_global = 0;
__device__ int d_num_mutants_global = 0;
__device__ double d_eliteCProb_global = 0.0;

#define CUDA_CHECK_HOST(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

BRKGA_GPU::BRKGA_GPU(const std::vector<Point3D_CPU>& input_boxes, const std::vector<Point3D_CPU>& input_bins_dims, int num_generations, int num_individuals, int num_elites, int num_mutants, double eliteCProb)
    : h_input_boxes(input_boxes), h_input_bins_dims(input_bins_dims),
      num_generations(num_generations), num_individuals(num_individuals),
      num_gene(2 * input_boxes.size()), num_elites(num_elites),
      num_mutants(num_mutants), eliteCProb(eliteCProb),
      d_population(nullptr), d_fitness_list(nullptr), d_elites(nullptr), d_non_elites(nullptr),
      best_fitness(1e9) {

    CUDA_CHECK_HOST(cudaMalloc(&d_population, (size_t)num_individuals * num_gene * sizeof(double)));
    CUDA_CHECK_HOST(cudaMalloc(&d_fitness_list, (size_t)num_individuals * sizeof(double)));
    CUDA_CHECK_HOST(cudaMalloc(&d_elites, (size_t)num_elites * sizeof(int)));
    CUDA_CHECK_HOST(cudaMalloc(&d_non_elites, (size_t)(num_individuals - num_elites) * sizeof(int)));

    int num_input_boxes = input_boxes.size();
    Point3D initial_bin_dim = Point3D(input_bins_dims[0].x, input_bins_dims[0].y, input_bins_dims[0].z);

    CUDA_CHECK_HOST(cudaMalloc(&d_input_boxes_global, num_input_boxes * sizeof(Point3D)));
    std::vector<Point3D> h_input_boxes_gpu(num_input_boxes);
    for(int i=0; i<num_input_boxes; ++i) {
        h_input_boxes_gpu[i] = Point3D(input_boxes[i].x, input_boxes[i].y, input_boxes[i].z);
    }
    CUDA_CHECK_HOST(cudaMemcpy(d_input_boxes_global, h_input_boxes_gpu.data(), num_input_boxes * sizeof(Point3D), cudaMemcpyHostToDevice));

    CUDA_CHECK_HOST(cudaMalloc(&d_all_BPS_global, (size_t)num_individuals * num_input_boxes * sizeof(int)));

    CUDA_CHECK_HOST(cudaMalloc(&d_pp_individuals_global, (size_t)num_individuals * sizeof(PlacementProcedure)));

    CUDA_CHECK_HOST(cudaMalloc(&d_all_bins_pool_global, (size_t)num_individuals * MAX_BINS_PER_INDIVIDUAL * sizeof(Bin)));

    initialize_population();

    cudaMemcpyToSymbol(d_num_individuals_global, &num_individuals, sizeof(int));
    cudaMemcpyToSymbol(d_num_gene_global, &num_gene, sizeof(int));
    cudaMemcpyToSymbol(d_num_elites_global, &num_elites, sizeof(int));
    cudaMemcpyToSymbol(d_num_mutants_global, &num_mutants, sizeof(int));
    cudaMemcpyToSymbol(d_eliteCProb_global, &eliteCProb, sizeof(double));
    cudaMemcpyToSymbol(d_population_global, &d_population, sizeof(double*));
    cudaMemcpyToSymbol(d_fitness_list_global, &d_fitness_list, sizeof(double*));
}

BRKGA_GPU::~BRKGA_GPU() {
    if (d_population) cudaFree(d_population);
    if (d_fitness_list) cudaFree(d_fitness_list);
    if (d_elites) cudaFree(d_elites);
    if (d_non_elites) cudaFree(d_non_elites);

    if (d_input_boxes_global) cudaFree(d_input_boxes_global);
    if (d_all_BPS_global) cudaFree(d_all_BPS_global);
    if (d_pp_individuals_global) cudaFree(d_pp_individuals_global);
    if (d_all_bins_pool_global) cudaFree(d_all_bins_pool_global);
}

void BRKGA_GPU::initialize_population() {
    std::vector<double> h_population(num_individuals * num_gene);
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < num_individuals * num_gene; ++i) {
        h_population[i] = distribution(generator);
    }

    CUDA_CHECK_HOST(cudaMemcpy(d_population, h_population.data(), (size_t)num_individuals * num_gene * sizeof(double), cudaMemcpyHostToDevice));
}

void BRKGA_GPU::partition() {
    thrust::device_vector<int> indices(num_individuals);
    thrust::sequence(indices.begin(), indices.end());

    thrust::device_ptr<double> d_fitness_ptr(d_fitness_list);
    thrust::device_vector<double> d_fitness_vector(d_fitness_ptr, d_fitness_ptr + num_individuals);

    thrust::sort_by_key(d_fitness_vector.begin(), d_fitness_vector.end(), indices.begin());

    thrust::copy(indices.begin(), indices.begin() + num_elites, thrust::device_ptr<int>(d_elites));

    thrust::copy(indices.begin() + num_elites, indices.end(), thrust::device_ptr<int>(d_non_elites));
}
__global__ void mutants_kernel(double* d_population, int num_individuals, int num_gene, int num_mutants) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_mutants * num_gene) return;
    curandState_t state;
    curand_init(clock() + idx, 0, 0, &state);

    int mutant_start_row = num_individuals - num_mutants;
    int row = mutant_start_row + idx / num_gene;
    int col = idx % num_gene;

    if (row < num_individuals) {
        d_population[row * num_gene + col] = curand_uniform(&state);
    }
}

__global__ void crossover_kernel(double* d_population, int* d_elites, int* d_non_elites, int num_individuals, int num_gene, int num_elites, int num_mutants, double eliteCProb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_offspring = num_individuals - num_elites - num_mutants;
    if (idx >= num_offspring * num_gene) return;

    curandState_t state;
    curand_init(clock() + idx, 0, 0, &state);

    int offspring_start_row = num_elites + num_mutants;
    int offspring_row = offspring_start_row + idx / num_gene;
    int col = idx % num_gene;

    int elite_idx = (int)(curand_uniform(&state) * num_elites);
    int non_elite_idx = (int)(curand_uniform(&state) * (num_individuals - num_elites));

    int elite_solution_idx = d_elites[elite_idx];
    int non_elite_solution_idx = d_non_elites[non_elite_idx];

    double elite_gene = d_population[elite_solution_idx * num_gene + col];
    double non_elite_gene = d_population[non_elite_solution_idx * num_gene + col];

    if (curand_uniform(&state) < eliteCProb) {
        d_population[offspring_row * num_gene + col] = elite_gene;
    } else {
        d_population[offspring_row * num_gene + col] = non_elite_gene;
    }
}

std::string BRKGA_GPU::fit(int patient) {
    calculate_fitness_cuda(h_input_boxes, h_input_bins_dims, d_population, d_fitness_list, num_individuals, num_gene,
                          d_input_boxes_global, d_all_BPS_global, d_pp_individuals_global, d_all_bins_pool_global);

    int best_iter = 0;
    std::cout << "-----------------------------PROCESSANDO-GERAÇÕES------------------------------\n";
    std::cout << "Geração->Fitness: \n\n";

    for (int g = 0; g < num_generations; ++g) {
        partition();

        int num_offspring = num_individuals - num_elites - num_mutants;
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_offspring * num_gene + threadsPerBlock - 1) / threadsPerBlock;
        crossover_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_population, d_elites, d_non_elites, num_individuals, num_gene, num_elites, num_mutants, eliteCProb);
        CUDA_CHECK_HOST(cudaGetLastError());

        blocksPerGrid = (num_mutants * num_gene + threadsPerBlock - 1) / threadsPerBlock;
        mutants_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_population, num_individuals, num_gene, num_mutants);
        CUDA_CHECK_HOST(cudaGetLastError());

        calculate_fitness_cuda(h_input_boxes, h_input_bins_dims, d_population, d_fitness_list, num_individuals, num_gene,
                              d_input_boxes_global, d_all_BPS_global, d_pp_individuals_global, d_all_bins_pool_global);

        thrust::device_ptr<double> d_fitness_ptr(d_fitness_list);
        double current_min_fitness = *thrust::min_element(d_fitness_ptr, d_fitness_ptr + num_individuals);

        if (current_min_fitness < best_fitness) {
            best_iter = g;
            best_fitness = current_min_fitness;
        }

        std::cout << std::setw(3) << g <<"->"<< std::setw(7) <<  best_fitness<< " | " << std::flush;
        
        if(!((g+1)%5)){
            std::cout << '\n';
        }

        if (g - best_iter > patient) {
            return "Bins utilizados: " + std::to_string(std::floor(best_fitness)) + "\nMelhor fitness: " + std::to_string(best_fitness);
        }
    }

    return "Bins utilizados: " + std::to_string(std::floor(best_fitness)) + "\nMelhor fitness: " + std::to_string(best_fitness);
}
