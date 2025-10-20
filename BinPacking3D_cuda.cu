#include "BinPacking3D.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

struct __align__(8) Point3D {
    int x, y, z;

    __host__ __device__ Point3D(int _x = 0, int _y = 0, int _z = 0) : x(_x), y(_y), z(_z) {}

    __host__ __device__ bool operator==(const Point3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    __host__ __device__ bool operator!=(const Point3D& other) const {
        return !(*this == other);
    }

    __host__ __device__ Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ bool operator<(const Point3D& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct __align__(16) EMS {
    Point3D min_corner;
    Point3D max_corner;

    __host__ __device__ EMS(Point3D min = Point3D(), Point3D max = Point3D()) : min_corner(min), max_corner(max) {}

    __host__ __device__ bool operator==(const EMS& other) const {
        return min_corner == other.min_corner && max_corner == other.max_corner;
    }
};

__host__ __device__ bool overlapped(EMS ems1, EMS ems2);
__host__ __device__ bool inscribed(EMS ems1, EMS ems2);
__host__ __device__ Point3D orient(Point3D box, int BO);
__host__ __device__ bool fit_in(Point3D box, EMS ems);

#define MAX_BOXES 1000
#define MAX_EMSS_PER_BIN 200
#define MAX_LOADED_ITEMS_PER_BIN MAX_BOXES
#define MAX_BINS_PER_INDIVIDUAL 500

struct __align__(8) Bin {
    Point3D dimensions;
    EMS EMSs[MAX_EMSS_PER_BIN];
    int num_EMSs;
    EMS loaded_items[MAX_LOADED_ITEMS_PER_BIN];
    int num_loaded_items;
};

struct __align__(8) PlacementProcedure {
    Bin* Bins_ptr;
    int num_bins_allocated;
    Point3D* boxes;
    int num_boxes;
    int* BPS;
    int num_BPS;
    double* VBO;
    int num_VBO;
    int num_opened_bins;
};

struct PlacementProcedure_setup {
    Point3D* boxes;
    int num_boxes;
    int* BPS_all;
    int num_BPS;
    double* VBO_all;
    int num_VBO;
    Point3D initial_bin_dim;
    Bin* all_bins_pool;
};

Point3D* d_input_boxes_global = nullptr;
int* d_all_BPS_global = nullptr;
double* d_all_VBO_global = nullptr;
PlacementProcedure* d_pp_individuals_global = nullptr;
double* d_fitness_list_global = nullptr;
Bin* d_all_bins_pool_global = nullptr;

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        if (d_input_boxes_global) cudaFree(d_input_boxes_global); \
        if (d_all_BPS_global) cudaFree(d_all_BPS_global); \
        if (d_all_VBO_global) cudaFree(d_all_VBO_global); \
        if (d_pp_individuals_global) cudaFree(d_pp_individuals_global); \
        if (d_fitness_list_global) cudaFree(d_fitness_list_global); \
        if (d_all_bins_pool_global) cudaFree(d_all_bins_pool_global); \
        return; \
    } \
}

__host__ __device__ bool overlapped(EMS ems1, EMS ems2) {
    return (ems1.max_corner.x > ems2.min_corner.x && ems1.max_corner.y > ems2.min_corner.y && ems1.max_corner.z > ems2.min_corner.z) &&
           (ems1.min_corner.x < ems2.max_corner.x && ems1.min_corner.y < ems2.max_corner.y && ems1.min_corner.z < ems2.max_corner.z);
}

__host__ __device__ bool inscribed(EMS ems1, EMS ems2) {
    return (ems2.min_corner.x <= ems1.min_corner.x && ems2.min_corner.y <= ems1.min_corner.y && ems2.min_corner.z <= ems1.min_corner.z) &&
           (ems1.max_corner.x <= ems2.max_corner.x && ems1.max_corner.y <= ems2.max_corner.y && ems1.max_corner.z <= ems2.max_corner.z);
}

__host__ __device__ Point3D orient(Point3D box, int BO) {
    int d = box.x, w = box.y, h = box.z;
    if (BO == 1) return Point3D(d, w, h);
    else if (BO == 2) return Point3D(d, h, w);
    else if (BO == 3) return Point3D(w, d, h);
    else if (BO == 4) return Point3D(w, h, d);
    else if (BO == 5) return Point3D(h, d, w);
    else if (BO == 6) return Point3D(h, w, d);
    return Point3D(0, 0, 0);
}

__host__ __device__ bool fit_in(Point3D box, EMS ems) {
    return (box.x <= (ems.max_corner.x - ems.min_corner.x)) &&
           (box.y <= (ems.max_corner.y - ems.min_corner.y)) &&
           (box.z <= (ems.max_corner.z - ems.min_corner.z));
}

__device__ void eliminate(EMS* ems_array, int* num_ems, EMS ems_to_remove) {
    int write_idx = 0;
    for (int i = 0; i < *num_ems; ++i) {
        if (!(ems_array[i] == ems_to_remove)) {
            ems_array[write_idx++] = ems_array[i];
        }
    }
    *num_ems = write_idx;
}

__device__ void bin_update(Bin* bin, Point3D box, EMS selected_EMS, int min_vol, int min_dim) {
    EMS ems_to_place = EMS(selected_EMS.min_corner, selected_EMS.min_corner + box);
    if (bin->num_loaded_items < MAX_LOADED_ITEMS_PER_BIN) {
        bin->loaded_items[bin->num_loaded_items++] = ems_to_place;
    }

    EMS current_EMSs_copy[MAX_EMSS_PER_BIN];
    int current_num_EMSs_copy = 0;
    for (int i = 0; i < bin->num_EMSs; ++i) {
        current_EMSs_copy[current_num_EMSs_copy++] = bin->EMSs[i];
    }

    for (int i = 0; i < current_num_EMSs_copy; ++i) {
        EMS EMS_item = current_EMSs_copy[i];
        if (overlapped(ems_to_place, EMS_item)) {
            eliminate(bin->EMSs, &bin->num_EMSs, EMS_item);

            int x1 = EMS_item.min_corner.x, y1 = EMS_item.min_corner.y, z1 = EMS_item.min_corner.z;
            int x2 = EMS_item.max_corner.x, y2 = EMS_item.max_corner.y, z2 = EMS_item.max_corner.z;
            int x3 = ems_to_place.min_corner.x, y3 = ems_to_place.min_corner.y, z3 = ems_to_place.min_corner.z;
            int x4 = ems_to_place.max_corner.x, y4 = ems_to_place.max_corner.y, z4 = ems_to_place.max_corner.z;

            EMS new_EMSs_candidates[6] = {
                EMS(Point3D(x1, y1, z1), Point3D(x3, y2, z2)),
                EMS(Point3D(x4, y1, z1), Point3D(x2, y2, z2)),
                EMS(Point3D(x1, y1, z1), Point3D(x2, y3, z2)),
                EMS(Point3D(x1, y4, z1), Point3D(x2, y2, z2)),
                EMS(Point3D(x1, y1, z1), Point3D(x2, y2, z3)),
                EMS(Point3D(x1, y1, z4), Point3D(x2, y2, z2))
            };

            for (int j = 0; j < 6; ++j) {
                EMS new_EMS = new_EMSs_candidates[j];
                Point3D new_box_dims = new_EMS.max_corner - new_EMS.min_corner;
                bool isValid = true;

                if (new_box_dims.x <= 0 || new_box_dims.y <= 0 || new_box_dims.z <= 0) {
                    isValid = false;
                }

                if (isValid) {
                    for (int k = 0; k < bin->num_EMSs; ++k) {
                        if (inscribed(new_EMS, bin->EMSs[k])) {
                            isValid = false;
                            break;
                        }
                    }
                }

                if (new_box_dims.x < min_dim || new_box_dims.y < min_dim || new_box_dims.z < min_dim) {
                    isValid = false;
                }

                if (static_cast<long long>(new_box_dims.x) * new_box_dims.y * new_box_dims.z < min_vol) {
                    isValid = false;
                }

                if (isValid) {
                    if (bin->num_EMSs < MAX_EMSS_PER_BIN) {
                        bin->EMSs[bin->num_EMSs++] = new_EMS;
                    }
                }
            }
        }
    }
}

__device__ double bin_load(Bin* bin) {
    long long loaded_volume = 0;
    for (int i = 0; i < bin->num_loaded_items; ++i) {
        EMS item = bin->loaded_items[i];
        loaded_volume += static_cast<long long>(item.max_corner.x - item.min_corner.x) *
                         (item.max_corner.y - item.min_corner.y) *
                         (item.max_corner.z - item.min_corner.z);
    }
    long long bin_volume = static_cast<long long>(bin->dimensions.x) * bin->dimensions.y * bin->dimensions.z;
    return static_cast<double>(loaded_volume) / bin_volume;
}

__device__ EMS DFTRC_2(Point3D box, Bin* bin) {
    double maxDist = -1.0;
    EMS selectedEMS = EMS();

    for (int i = 0; i < bin->num_EMSs; ++i) {
        EMS EMS_item = bin->EMSs[i];
        int D_bin = bin->dimensions.x;
        int W_bin = bin->dimensions.y;
        int H_bin = bin->dimensions.z;

        for (int direction = 1; direction <= 6; ++direction) {
            Point3D oriented_box = orient(box, direction);
            if (fit_in(oriented_box, EMS_item)) {
                int x = EMS_item.min_corner.x;
                int y = EMS_item.min_corner.y;
                int z = EMS_item.min_corner.z;
                
                float distance = powf((float)D_bin - x - oriented_box.x, 2.0f) +
                                 powf((float)W_bin - y - oriented_box.y, 2.0f) +
                                 powf((float)H_bin - z - oriented_box.z, 2.0f);

                if (distance > maxDist) {
                    maxDist = distance;
                    selectedEMS = EMS_item;
                }
            }
        }
    }
    return selectedEMS;
}

__device__ int select_box_orientation(double VBO_val, Point3D box, EMS ems) {
    int BOs[6];
    int num_BOs = 0;
    for (int direction = 1; direction <= 6; ++direction) {
        if (fit_in(orient(box, direction), ems)) {
            BOs[num_BOs++] = direction;
        }
    }
    if (num_BOs == 0) return 0;

    int selected_idx = static_cast<int>(ceil(VBO_val * num_BOs)) - 1;
    if (selected_idx < 0) selected_idx = 0;
    if (selected_idx >= num_BOs) selected_idx = num_BOs - 1;

    return BOs[selected_idx];
}

__device__ void elimination_rule(Point3D* remaining_boxes, int num_remaining_boxes, int* min_vol, int* min_dim) {
    if (num_remaining_boxes == 0) {
        *min_vol = 0;
        *min_dim = 0;
        return;
    }

    *min_vol = 999999999;
    *min_dim = 9999;
    for (int i = 0; i < num_remaining_boxes; ++i) {
        Point3D box = remaining_boxes[i];
        int dim = min(min(box.x, box.y), box.z);
        if (dim < *min_dim) {
            *min_dim = dim;
        }
        long long vol = static_cast<long long>(box.x) * box.y * box.z;
        if (vol < *min_vol) {
            *min_vol = static_cast<int>(vol);
        }
    }
}

__device__ double evaluate(PlacementProcedure* pp) {
    double leastLoad = 1.0;
    for (int k = 0; k < pp->num_opened_bins; ++k) {
        double load = bin_load(&pp->Bins_ptr[k]);
        if (load < leastLoad) {
            leastLoad = load;
        }
    }
    return static_cast<double>(pp->num_opened_bins) + leastLoad;
}

__global__ void initialize_placement_kernel(PlacementProcedure* d_pp_individuals, PlacementProcedure_setup setup, int total_individuals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_individuals) return;

    PlacementProcedure* pp = &d_pp_individuals[idx];

    pp->num_bins_allocated = MAX_BINS_PER_INDIVIDUAL;
    pp->num_opened_bins = 1;
    pp->boxes = setup.boxes;
    pp->num_boxes = setup.num_boxes;

    pp->BPS = setup.BPS_all + idx * setup.num_BPS;
    pp->num_BPS = setup.num_BPS;
    pp->VBO = setup.VBO_all + idx * setup.num_VBO;
    pp->num_VBO = setup.num_VBO;
    pp->Bins_ptr = setup.all_bins_pool + idx * MAX_BINS_PER_INDIVIDUAL;

    pp->Bins_ptr[0].dimensions = setup.initial_bin_dim;
    pp->Bins_ptr[0].num_EMSs = 1;
    pp->Bins_ptr[0].EMSs[0] = EMS(Point3D(0,0,0), setup.initial_bin_dim);
    pp->Bins_ptr[0].num_loaded_items = 0;
}

__global__ void placement_kernel(PlacementProcedure* d_pp_individuals, double* d_fitness_output, int total_individuals) {
    int individual_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (individual_idx >= total_individuals) return;

    PlacementProcedure* pp = &d_pp_individuals[individual_idx];

    Point3D items_sorted[MAX_BOXES];
    for (int i = 0; i < pp->num_BPS; ++i) {
        items_sorted[i] = pp->boxes[pp->BPS[i]];
    }

    for (int i = 0; i < pp->num_BPS; ++i) {
        Point3D box = items_sorted[i];
        int selected_bin_idx = -1;
        EMS selected_EMS = EMS();

        for (int k = 0; k < pp->num_opened_bins; ++k) {
            EMS ems = DFTRC_2(box, &pp->Bins_ptr[k]);
            if (ems.min_corner.x != 0 || ems.min_corner.y != 0 || ems.min_corner.z != 0 ||
                ems.max_corner.x != 0 || ems.max_corner.y != 0 || ems.max_corner.z != 0) {
                selected_bin_idx = k;
                selected_EMS = ems;
                break;
            }
        }

        if (selected_bin_idx == -1) {
            if (pp->num_opened_bins < MAX_BINS_PER_INDIVIDUAL) {
                selected_bin_idx = pp->num_opened_bins;
                pp->num_opened_bins++;
                pp->Bins_ptr[selected_bin_idx].dimensions = pp->Bins_ptr[0].dimensions;
                pp->Bins_ptr[selected_bin_idx].num_EMSs = 1;
                pp->Bins_ptr[selected_bin_idx].EMSs[0] = EMS(Point3D(0,0,0), pp->Bins_ptr[0].dimensions);
                pp->Bins_ptr[selected_bin_idx].num_loaded_items = 0;
                selected_EMS = pp->Bins_ptr[selected_bin_idx].EMSs[0];
            } else {
                 d_fitness_output[individual_idx] = 1e9;
                 return;
            }
        }

        int BO = select_box_orientation(pp->VBO[i], box, selected_EMS);

        Point3D remaining_boxes_slice[MAX_BOXES];
        int num_remaining_boxes_slice = 0;
        if (i + 1 < pp->num_BPS) {
            for (int j = i + 1; j < pp->num_BPS; ++j) {
                remaining_boxes_slice[num_remaining_boxes_slice++] = items_sorted[j];
            }
        }
        int min_vol, min_dim;
        elimination_rule(remaining_boxes_slice, num_remaining_boxes_slice, &min_vol, &min_dim);

        bin_update(&pp->Bins_ptr[selected_bin_idx], orient(box, BO), selected_EMS, min_vol, min_dim);
    }
    d_fitness_output[individual_idx] = evaluate(pp);
}

void calculate_fitness(const std::vector<Point3D_CPU>& input_boxes, const std::vector<Point3D_CPU>& input_bins_dims,
                           const std::vector<std::vector<double>>& population, std::vector<double>& fitness_list) {
    int num_individuals = population.size();
    int num_gene = population[0].size();
    int num_input_boxes = input_boxes.size();


    const int BATCH_SIZE = 100;
    int num_batches = (num_individuals + BATCH_SIZE - 1) / BATCH_SIZE;

    fitness_list.resize(num_individuals);

    Point3D* d_input_boxes = nullptr;
    cudaError_t err = cudaMalloc(&d_input_boxes, num_input_boxes * sizeof(Point3D));
    CUDA_CHECK(err);
    std::vector<Point3D> h_input_boxes(num_input_boxes);
    for(int i=0; i<num_input_boxes; ++i) {
        h_input_boxes[i] = Point3D(input_boxes[i].x, input_boxes[i].y, input_boxes[i].z);
    }
    err = cudaMemcpy(d_input_boxes, h_input_boxes.data(), num_input_boxes * sizeof(Point3D), cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    int* d_batch_BPS = nullptr;
    double* d_batch_VBO = nullptr;
    PlacementProcedure* d_batch_pp_individuals = nullptr;
    double* d_batch_fitness_list = nullptr;
    Bin* d_batch_bins_pool = nullptr;

    size_t max_batch_bps_size = (size_t)BATCH_SIZE * num_input_boxes * sizeof(int);
    size_t max_batch_vbo_size = (size_t)BATCH_SIZE * (num_gene - num_input_boxes) * sizeof(double);
    size_t max_batch_pp_individuals_size = (size_t)BATCH_SIZE * sizeof(PlacementProcedure);
    size_t max_batch_fitness_list_size = (size_t)BATCH_SIZE * sizeof(double);
    size_t max_batch_bins_pool_size = (size_t)BATCH_SIZE * MAX_BINS_PER_INDIVIDUAL * sizeof(Bin);

    err = cudaMalloc(&d_batch_BPS, max_batch_bps_size);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_batch_VBO, max_batch_vbo_size);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_batch_pp_individuals, max_batch_pp_individuals_size);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_batch_fitness_list, max_batch_fitness_list_size);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_batch_bins_pool, max_batch_bins_pool_size);
    CUDA_CHECK(err);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        int current_batch_start_idx = batch_idx * BATCH_SIZE;
        int current_batch_size = std::min(BATCH_SIZE, num_individuals - current_batch_start_idx);

        std::vector<int> h_batch_BPS(current_batch_size * num_input_boxes);
        std::vector<double> h_batch_VBO(current_batch_size * (num_gene - num_input_boxes));

        for (int i = 0; i < current_batch_size; ++i) {
            const std::vector<double>& solution = population[current_batch_start_idx + i];
            std::vector<std::pair<double, int>> indexed_solution_bps(num_input_boxes);
            for (int j = 0; j < num_input_boxes; ++j) {
                indexed_solution_bps[j] = {solution[j], j};
            }
            std::sort(indexed_solution_bps.begin(), indexed_solution_bps.end());
            
            for (int j = 0; j < num_input_boxes; ++j) {
                h_batch_BPS[i * num_input_boxes + j] = indexed_solution_bps[j].second;
            }

            for (int j = 0; j < (num_gene - num_input_boxes); ++j) {
                h_batch_VBO[i * (num_gene - num_input_boxes) + j] = solution[num_input_boxes + j];
            }
        }

        err = cudaMemcpy(d_batch_BPS, h_batch_BPS.data(), current_batch_size * num_input_boxes * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_CHECK(err);
        err = cudaMemcpy(d_batch_VBO, h_batch_VBO.data(), current_batch_size * (num_gene - num_input_boxes) * sizeof(double), cudaMemcpyHostToDevice);
        CUDA_CHECK(err);

        PlacementProcedure_setup setup;
        setup.boxes = d_input_boxes;
        setup.num_boxes = num_input_boxes;
        setup.BPS_all = d_batch_BPS;
        setup.num_BPS = num_input_boxes;
        setup.VBO_all = d_batch_VBO;
        setup.num_VBO = num_gene - num_input_boxes;
        setup.initial_bin_dim = Point3D(input_bins_dims[0].x, input_bins_dims[0].y, input_bins_dims[0].z);
        setup.all_bins_pool = d_batch_bins_pool;

        int threadsPerBlock = 256;
        int blocksPerGrid = (current_batch_size + threadsPerBlock - 1) / threadsPerBlock;

        initialize_placement_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_batch_pp_individuals, setup, current_batch_size);
        err = cudaGetLastError();
        CUDA_CHECK(err);
        err = cudaDeviceSynchronize();
        CUDA_CHECK(err);

        placement_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_batch_pp_individuals, d_batch_fitness_list, current_batch_size);
        err = cudaGetLastError();
        CUDA_CHECK(err);
        err = cudaDeviceSynchronize();
        CUDA_CHECK(err);

        err = cudaMemcpy(fitness_list.data() + current_batch_start_idx, d_batch_fitness_list, current_batch_size * sizeof(double), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);
    }

    cudaFree(d_batch_BPS);
    cudaFree(d_batch_VBO);
    cudaFree(d_batch_pp_individuals);
    cudaFree(d_batch_fitness_list);
    cudaFree(d_batch_bins_pool);
    cudaFree(d_input_boxes);
}

