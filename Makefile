NVCC = nvcc
CXX = g++

STD_FLAG_HOST = -std=c++14
STD_FLAG_NVCC = -std=c++14

NVCCFLAGS = -arch=sm_75 $(STD_FLAG_NVCC) --expt-relaxed-constexpr --expt-extended-lambda -ccbin g++-10

CXXFLAGS = $(STD_FLAG_HOST) -Wall -Wextra -O2

TARGET = BRKGA

CPP_SRCS = main.cpp BinPacking3D.cpp
CU_SRCS = BinPacking3D_cuda.cu BRKGA_GPU.cu

CPP_OBJS = $(CPP_SRCS:.cpp=.o)
CU_OBJS = $(CU_SRCS:.cu=.o)
OBJS = $(CPP_OBJS) $(CU_OBJS)

LDFLAGS = -lcudart

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
	@echo "Limpeza concluída!"
