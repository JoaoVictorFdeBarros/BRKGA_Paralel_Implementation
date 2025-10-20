
#include "BinPacking3D.hpp"
#include <random>
#include <chrono>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <iomanip>

namespace InstanceGenerator {
    unsigned int _h48, _l48;

    int ur(int lb, int ub) {
        static std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<int> distribution(static_cast<int>(lb), static_cast<int>(ub));
        int value = distribution(generator);
        return static_cast<int>(value >= 1 ? value : 1);
    }

    void generateInstances(int N, std::vector<Point3D_CPU>& pqr, std::vector<Point3D_CPU>& LWH,int type) {
        Point3D_CPU V = {100, 100, 100};

        switch (type)
        {
        case 6:
            V= {10,10,10};
            break;
        case 7:
            V = {40,40,40};
            break;
        default:
            break;
        }

        pqr.clear();
        LWH.clear();
        
        int Wmax = 0;
        int Wmin = 0;
        int Hmax = 0;
        int Hmin = 0;
        int Dmax = 0;
        int Dmin = 0;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> randomize(0,1);
        std::uniform_int_distribution<> randomType(1,5);

        for (int i = 0; i < N; ++i) {
            int newType = type;
            if(randomize(gen) && newType <= 5){
                newType = randomType(gen);
            }

            switch (newType)
            {
            case 1:
                Wmax = V.x/2;
                Wmin = 1;
                Hmax = V.y;
                Hmin = 2*V.y/3;
                Dmax = V.z;
                Dmin = 2*V.z/3;
                break;
            case 2:
                Wmax = V.x;
                Wmin = 2*V.x/3;
                Hmax = V.y/2;
                Hmin = 1;
                Dmax = V.z;
                Dmin = 2*V.z/3;
                break;
            case 3:
                Wmax = V.x;
                Wmin = 2*V.x/3;
                Hmax = V.y;
                Hmin = 2*V.y/3;
                Dmax = V.z/2;
                Dmin = 1;
                break;
            case 4:
                Wmax = V.x;
                Wmin = V.x/2;
                Hmax = V.y;
                Hmin = V.y/2;
                Dmax = V.z;
                Dmin = V.z/2;
                break;
            case 5:
                Wmax = V.x/2;
                Wmin = 1;
                Hmax = V.y/2;
                Hmin = 1;
                Dmax = V.z/2;
                Dmin = 1;
                break;
            case 6:
                Wmax = 10;
                Wmin = 1;
                Hmax = 10;
                Hmin = 1;
                Dmax = 10;
                Dmin = 1;
                break; 
            case 7:
                Wmax = 35;
                Wmin = 1;
                Hmax = 35;
                Hmin = 1;
                Dmax = 35;
                Dmin = 1;
                break;           
            case 8:
                Wmax = 100;
                Wmin = 1;
                Hmax = 100;
                Hmin = 1;
                Dmax = 100;
                Dmin = 1;
                break;  
            default:
                break;
            }

            Point3D_CPU new_box = {ur(Wmin, Wmax), ur(Hmin, Hmax), ur(Dmin, Dmax)};
            pqr.push_back(new_box);
        }
        for (int i = 0; i < 10000; ++i) {
            LWH.push_back(V);
        }
    }
}


BRKGA::BRKGA(const std::vector<Point3D_CPU>& input_boxes, const std::vector<Point3D_CPU>& input_bins_dims, int num_generations, int num_individuals, int num_elites, int num_mutants, double eliteCProb)
    : input_boxes(input_boxes), input_bins_dims(input_bins_dims),
      N(input_boxes.size()), num_generations(num_generations),
      num_individuals(num_individuals), num_gene(2 * input_boxes.size()),
      num_elites(num_elites), num_mutants(num_mutants), eliteCProb(eliteCProb),
      used_bins(-1), best_fitness(-1) {}

void BRKGA::partition(const std::vector<std::vector<double>>& population, const std::vector<double>& fitness_list, 
                      std::vector<std::vector<double>>& elites, std::vector<std::vector<double>>& non_elites, 
                      std::vector<double>& elite_fitness_list) {
    
    std::vector<std::pair<double, int>> indexed_fitness(fitness_list.size());
    for (size_t i = 0; i < fitness_list.size(); ++i) {
        indexed_fitness[i] = {fitness_list[i], i};
    }
    std::sort(indexed_fitness.begin(), indexed_fitness.end());

    elites.clear();
    non_elites.clear();
    elite_fitness_list.clear();

    for (int i = 0; i < num_elites; ++i) {
        elites.push_back(population[indexed_fitness[i].second]);
        elite_fitness_list.push_back(indexed_fitness[i].first);
    }

    for (size_t i = num_elites; i < population.size(); ++i) {
        non_elites.push_back(population[indexed_fitness[i].second]);
    }
}

std::vector<double> BRKGA::crossover(const std::vector<double>& elite, const std::vector<double>& non_elite) {
    std::vector<double> offspring(num_gene);
    static std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < num_gene; ++i) {
        if (distribution(generator) < eliteCProb) {
            offspring[i] = elite[i];
        } else {
            offspring[i] = non_elite[i];
        }
    }
    return offspring;
}

std::vector<std::vector<double>> BRKGA::mating(const std::vector<std::vector<double>>& elites, const std::vector<std::vector<double>>& non_elites) {
    int num_offspring = num_individuals - num_elites - num_mutants;
    std::vector<std::vector<double>> offsprings;
    offsprings.reserve(num_offspring);

    static std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> elite_dist(0, elites.size() - 1);
    std::uniform_int_distribution<int> non_elite_dist(0, non_elites.size() - 1);

    for (int i = 0; i < num_offspring; ++i) {
        const auto& elite_parent = elites[elite_dist(generator)];
        const auto& non_elite_parent = non_elites[non_elite_dist(generator)];
        offsprings.push_back(crossover(elite_parent, non_elite_parent));
    }
    return offsprings;
}

std::vector<std::vector<double>> BRKGA::mutants() {
    std::vector<std::vector<double>> new_mutants(num_mutants, std::vector<double>(num_gene));
    static std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < num_mutants; ++i) {
        for (int j = 0; j < num_gene; ++j) {
            new_mutants[i][j] = distribution(generator);
        }
    }
    return new_mutants;
}

std::string BRKGA::fit(int patient) {
    std::vector<std::vector<double>> population(num_individuals, std::vector<double>(num_gene));
    static std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < num_individuals; ++i) {
        for (int j = 0; j < num_gene; ++j) {
            population[i][j] = distribution(generator);
        }
    }
    std::vector<double> fitness_list;
    calculate_fitness(input_boxes, input_bins_dims, population, fitness_list);

    double max_fitness = -1.0;
    if (!fitness_list.empty()) {
        max_fitness = *std::max_element(fitness_list.begin(), fitness_list.end());
    }
    std::cout << "\nFitness inicial: " << max_fitness << "\n\n";
    

    best_fitness = 100000;
    if (!fitness_list.empty()) {
        best_fitness = *std::min_element(fitness_list.begin(), fitness_list.end());
        best_solution = population[std::min_element(fitness_list.begin(), fitness_list.end()) - fitness_list.begin()];
    }
    history_min.push_back(best_fitness);
    history_mean.push_back(std::accumulate(fitness_list.begin(), fitness_list.end(), 0.0) / fitness_list.size());

    int best_iter = 0;
    std::cout << "-----------------------------PROCESSANDO-GERAÇÕES------------------------------\n";
    std::cout << "Geração->Fitness: \n\n";
    for (int g = 0; g < num_generations; ++g) {
        if (g - best_iter > patient) {
            used_bins = std::floor(best_fitness);
            return "feasible";
        }

        std::vector<std::vector<double>> elites, non_elites;
        std::vector<double> elite_fitness_list;
        partition(population, fitness_list, elites, non_elites, elite_fitness_list);

        std::vector<std::vector<double>> offsprings = mating(elites, non_elites);

        std::vector<std::vector<double>> new_mutants = mutants();

        std::vector<std::vector<double>> next_population;
        next_population.reserve(elites.size() + new_mutants.size() + offsprings.size());
        next_population.insert(next_population.end(), elites.begin(), elites.end());
        next_population.insert(next_population.end(), new_mutants.begin(), new_mutants.end());
        next_population.insert(next_population.end(), offsprings.begin(), offsprings.end());
        
        population = next_population;
        calculate_fitness(input_boxes, input_bins_dims, population, fitness_list);

        double current_min_fitness = 100000;
        if (!fitness_list.empty()) {
            current_min_fitness = *std::min_element(fitness_list.begin(), fitness_list.end());
        }

        if (current_min_fitness < best_fitness) {
            best_iter = g;
            best_fitness = current_min_fitness;
            best_solution = population[std::min_element(fitness_list.begin(), fitness_list.end()) - fitness_list.begin()];
        }

        history_min.push_back(current_min_fitness);
        history_mean.push_back(std::accumulate(fitness_list.begin(), fitness_list.end(), 0.0) / fitness_list.size());

        std::cout << std::setw(3) << g <<"->"<< std::setw(7) <<  best_fitness<< " | " << std::flush;
        
        if(!((g+1)%5)){
            std::cout << '\n';
        }
    }

    used_bins = std::floor(best_fitness);
    return "feasible";
}


