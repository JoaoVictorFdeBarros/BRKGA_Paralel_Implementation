CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g

NVCC = nvcc
NVCCFLAGS = -std=c++14 -arch=sm_75

TARGET = BRKGA
SOURCES_CPP = main.cpp BinPacking3D.cpp
SOURCES_CUDA = BinPacking3D_cuda.cu

OBJECTS_CPP = $(SOURCES_CPP:.cpp=.o)
OBJECTS_CUDA = $(SOURCES_CUDA:.cu=.o)

.PHONY: all clean

all: $(TARGET)

BRKGA: $(OBJECTS_CPP) BinPacking3D_cuda.o
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS_CPP) BinPacking3D_cuda.o -L/usr/local/cuda/lib64 -lcudart

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS_CPP) $(OBJECTS_CUDA) $(TARGET)
	@echo "Limpeza concluída!"

