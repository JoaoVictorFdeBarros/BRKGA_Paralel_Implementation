
#ifndef BINPACKING3D_HPP
#define BINPACKING3D_HPP

#include <vector>
#include <array>

#include <iostream>

struct Point3D_CPU {
    int x, y, z;

    bool operator==(const Point3D_CPU& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const Point3D_CPU& other) const {
        return !(*this == other);
    }

    Point3D_CPU operator+(const Point3D_CPU& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Point3D_CPU operator-(const Point3D_CPU& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    bool operator<(const Point3D_CPU& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};
class BRKGA {
public:
    std::vector<Point3D_CPU> input_boxes;
    std::vector<Point3D_CPU> input_bins_dims;
    int N;

    int num_generations;
    int num_individuals;
    int num_gene;
    int num_elites;
    int num_mutants;
    double eliteCProb;

    double used_bins;
    double best_fitness;
    std::vector<double> best_solution;
    std::vector<double> history_min;
    std::vector<double> history_mean;

    BRKGA(const std::vector<Point3D_CPU>& input_boxes, const std::vector<Point3D_CPU>& input_bins_dims, int num_generations = 200, int num_individuals = 120, int num_elites = 12, int num_mutants = 18, double eliteCProb = 0.7);

    void partition(const std::vector<std::vector<double>>& population, const std::vector<double>& fitness_list, std::vector<std::vector<double>>& elites, std::vector<std::vector<double>>& non_elites, std::vector<double>& elite_fitness_list);
    std::vector<double> crossover(const std::vector<double>& elite, const std::vector<double>& non_elite);
    std::vector<std::vector<double>> mating(const std::vector<std::vector<double>>& elites, const std::vector<std::vector<double>>& non_elites);
    std::vector<std::vector<double>> mutants();
    std::string fit(int patient = 4);
};

namespace InstanceGenerator {
    int ur(int lb, int ub);

    void generateInstances(int N, std::vector<Point3D_CPU>& pqr, std::vector<Point3D_CPU>& LWH,int type);
}

#endif

void calculate_fitness(const std::vector<Point3D_CPU>& input_boxes, const std::vector<Point3D_CPU>& input_bins_dims,
                           const std::vector<std::vector<double>>& population, std::vector<double>& fitness_list);

