CXX      = g++
CXXFLAGS = -std=c++14 -Ofast
LDFLAGS  = -L/opt/cuda-8.0/lib64 
LDLIBS   = -lboost_program_options -lcuda -lcudart
NVFLAGS  = -std=c++11 -arch=sm_61
NVLDFLAGS= 
NVLDLIBS =

all: kmeans

.PHONY: add
add: format
	git add .

.PHONY: format
format: 
	clang-format -i *.c++ *.cu *.h

.PHONY: clean
clean:
	-rm -f *.o
	-rm -f *.gch
	-rm -f vgcore.*
	-rm -f kmeans

kmeans: kmeans.o point.o km_cpu.o km_cuda.o
	$(CXX) $(LDFLAGS) kmeans.o point.o km_cpu.o km_cuda.o -o kmeans $(LDLIBS)

km_cuda.o: km_cuda.cu km_cuda.cuh km_cuda.h
	nvcc $(NVFLAGS) $(NVLDFLAGS) km_cuda.cu -c $(NVLDLIBS)

kmeans.o: kmeans.c++ KMParams.h point.h km_cpu.h km_cuda.h
	$(CXX) $(CXXFLAGS) kmeans.c++ -c

point.o: point.c++ point.h
	$(CXX) $(CXXFLAGS) point.c++ -c

km_cpu.o: km_cpu.c++ km_cpu.h KMParams.h point.h
	$(CXX) $(CXXFLAGS) km_cpu.c++ -c
