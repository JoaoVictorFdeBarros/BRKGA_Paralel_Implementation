
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


namespace InstanceGenerator {
    int ur(int lb, int ub);

    void generateInstances(int N, std::vector<Point3D_CPU>& pqr, std::vector<Point3D_CPU>& LWH,int type);
}

#endif

