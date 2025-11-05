
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





