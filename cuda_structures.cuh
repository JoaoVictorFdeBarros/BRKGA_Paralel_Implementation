#ifndef CUDA_STRUCTURES_CUH
#define CUDA_STRUCTURES_CUH

#include <cuda_runtime.h>

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

#define MAX_BOXES 1000
#define MAX_EMSS_PER_BIN 200
#define MAX_LOADED_ITEMS_PER_BIN MAX_BOXES
#define MAX_BINS_PER_INDIVIDUAL 50

struct Bin {
    Point3D dimensions;
    EMS EMSs[MAX_EMSS_PER_BIN];
    int num_EMSs;
    EMS loaded_items[MAX_LOADED_ITEMS_PER_BIN];
    int num_loaded_items;
};

struct PlacementProcedure {
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

#endif
