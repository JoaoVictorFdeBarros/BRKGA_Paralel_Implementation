#ifndef BINPACKING3D_CUDA_HPP
#define BINPACKING3D_CUDA_HPP

#include "BinPacking3D.hpp"
#include <vector>

struct Point3D;
struct Bin;
struct PlacementProcedure;

void calculate_fitness_cuda(const std::vector<Point3D_CPU>& input_boxes, 
                            const std::vector<Point3D_CPU>& input_bins_dims,
                            double* d_population, 
                            double* d_fitness_list, 
                            int num_individuals, 
                            int num_gene,
                            Point3D* d_input_boxes_global,
                            int* d_all_BPS_global,
                            PlacementProcedure* d_pp_individuals_global,
                            Bin* d_all_bins_pool_global);

#endif
