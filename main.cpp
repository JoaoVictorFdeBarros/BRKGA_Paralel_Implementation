#include "BinPacking3D.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

void print_usage() {
std::cout  << " \n-----------------------------------Parâmetros----------------------------------\n";
std::cout << "-n <num>: Número de caixas (padrão: 100)\n";
std::cout << "-g <num>: Número de gerações do BRKGA (padrão: 200)\n";
std::cout << "-p <num>: Tamanho da população (padrão: 30*n)\n";
std::cout << "-e <num>: Número de elites (padrão: 0.1*p)\n";
std::cout << "-mut <num>: Número de mutantes (padrão: 0.15*p)\n";
std::cout << "-prob <float>: Probabilidade de crossover elite (padrão: 0.7)\n";
std::cout << "-patience <num>: Paciência para early stopping (padrão: 10)\n";
std::cout << "-t <num>: define o tipo das instancias [1-8] (ver artigo)\n";
std::cout << "-h: Ajuda.\n";
    std::cout << "-------------------------------------------------------------------------------\n\n";
}

Point3D_CPU parse_dimensions(const std::string& dim_str) {
    std::stringstream ss(dim_str);
    std::string item;
    std::vector<int> dims;
    
    while (std::getline(ss, item, ',')) {
        dims.push_back(std::stoi(item));
    }
    
    if (dims.size() != 3) {
        throw std::invalid_argument("Dimensões devem ser especificadas como x,y,z");
    }
    
    return {dims[0], dims[1], dims[2]};
}

void print_parameters(int N, int num_generations, int num_individuals, int num_elites, int num_mutants, double eliteCProb, int patience, int type) {
    std::cout << "\n\n----------------------------PARÂMETROS-DO-ALGORITMO----------------------------\n";
    std::cout << "Tipo de instancias: " << type<<"\n";
    std::cout << "Número de caixas: " << N << "\n";
    std::cout << "Número de gerações: " << num_generations << "\n";
    std::cout << "Tamanho da população: " << num_individuals << "\n";
    std::cout << "Número de elites: " << num_elites << "\n";
    std::cout << "Número de mutantes: " << num_mutants << "\n";
    std::cout << "Probabilidade de crossover elite: " << eliteCProb << "\n";
    std::cout << "Paciência para early stopping: " << patience << "\n";
}

void print_instances(const std::vector<Point3D_CPU>& boxes) {

    std::cout << "\n------------------------------INSTÂNCIAS-GERADAS-------------------------------\n";
    for (size_t i = 0; i < boxes.size(); ++i) {
        std::cout << "("<< std::setw(3) << boxes[i].x << "," << std::setw(3) << boxes[i].y << "," << std::setw(3) << boxes[i].z << ") | ";
        if(!((i+1)%5)){
            std::cout << '\n';
        }
    }
    std::cout << "-------------------------------------------------------------------------------\n";
}

void print_results(const BRKGA& algorithm) {
    std::cout << "\n\n----------------------------------RESULTADOS-----------------------------------\n";
    std::cout << "Bins utilizados: " << algorithm.used_bins << "\n";
    std::cout << "Melhor fitness:  " << std::fixed << std::setprecision(6) << algorithm.best_fitness << "\n";
    
    if (!algorithm.history_min.empty()) {
        std::cout << "Melhoria:    " << std::fixed << std::setprecision(6) << (algorithm.history_min[0] - algorithm.history_min.back()) << "\n";
    }
}

int main(int argc, char* argv[]) {

    int N = 100;
    int num_generations = 200;
    int num_individuals = 30*N;
    int num_elites = num_individuals*0.1;
    int num_mutants = num_individuals*0.15;
    double eliteCProb = 0.7;
    int patience = 10;
    int type = 1;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "-n" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
            num_individuals = 30*N;
            num_elites = num_individuals*0.1;
            num_mutants = num_individuals*0.15;
        } else if (arg == "-g" && i + 1 < argc) {
            num_generations = std::stoi(argv[++i]);
        } else if (arg == "-p" && i + 1 < argc) {
            num_individuals = std::stoi(argv[++i]);
        } else if (arg == "-e" && i + 1 < argc) {
            num_elites = std::stoi(argv[++i]);
        } else if (arg == "-mut" && i + 1 < argc) {
            num_mutants = std::stoi(argv[++i]);
        } else if (arg == "-prob" && i + 1 < argc) {
            eliteCProb = std::stod(argv[++i]);
        } else if (arg == "-patience" && i + 1 < argc) {
            patience = std::stoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            type = std::stoi(argv[++i]);
        }else {
            std::cerr << "Argumento desconhecido: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    if (N <= 0) {
        std::cerr << "Erro: Todos os valores numéricos devem ser positivos.\n";
        return 1;
    }
    
    if (num_elites + num_mutants >= num_individuals) {
        std::cout << " num_elites: " << num_elites << "\n num_mutants: " << num_mutants << "\n num_individuals: " << num_individuals << "\n\n";
        std::cerr << "Erro: A soma de elites e mutantes deve ser menor que o tamanho da população.\n";
        return 1;
    }
    
    print_parameters(N, num_generations, num_individuals, num_elites, num_mutants, eliteCProb, patience,type);

    std::vector<Point3D_CPU> boxes, bins_dims;
    InstanceGenerator::generateInstances(N, boxes, bins_dims,type);

    print_instances(boxes);

    BRKGA algorithm(boxes, bins_dims, num_generations, num_individuals, num_elites, num_mutants, eliteCProb);
    std::string result = algorithm.fit(patience);

    print_results(algorithm);

    // if (!algorithm.best_solution.empty()) {
    //     std::cout << "----------------------------------USO-DOS-BINS-----------------------------------\n";
    //     PlacementProcedure final_placement(boxes, bins_dims, algorithm.best_solution);

    //     for (int i = 0; i < final_placement.num_opened_bins; ++i) {
    //         double load = final_placement.Bins[i].load();
    //         std::cout << std::fixed << std::setprecision(2) << (load * 100) << "% | ";
    //         if(!((i+1)%5)){
    //             std::cout << '\n';
    //         }
    //     }
    //     std::cout << "\n-------------------------------------------------------------------------------\n\n";
    // }

    return 0;
}

