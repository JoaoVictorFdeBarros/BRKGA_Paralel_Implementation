#ifndef BRKGA_GPU_HPP
#define BRKGA_GPU_HPP

#include "BinPacking3D.hpp"
#include "BinPacking3D_cuda.hpp"

class BRKGA_GPU {
public:
    BRKGA_GPU(const std::vector<Point3D_CPU>& input_boxes, const std::vector<Point3D_CPU>& input_bins_dims, int num_generations, int num_individuals, int num_elites, int num_mutants, double eliteCProb);
    ~BRKGA_GPU();

    std::string fit(int patient = 4);

private:
    int num_generations;
    int num_individuals;
    int num_gene;
    int num_elites;
    int num_mutants;
    double eliteCProb;

    std::vector<Point3D_CPU> h_input_boxes;
    std::vector<Point3D_CPU> h_input_bins_dims;

    double* d_population;
    double* d_fitness_list;
    int* d_elites;
    int* d_non_elites;

    double best_fitness;
    std::vector<double> best_solution;

    Point3D* d_input_boxes_global;
    int* d_all_BPS_global;
    PlacementProcedure* d_pp_individuals_global;
    Bin* d_all_bins_pool_global;

    void initialize_population();
    void partition();
};

#endif
