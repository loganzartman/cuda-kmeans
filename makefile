HOST=$(shell hostname)

ifneq ($(wildcard /opt/cuda-8.0/.),)
	CUDADIR = /opt/cuda-8.0/lib64  # UTCS lab machines
else
	CUDADIR = /opt/cuda/lib64      # alternative
endif

ifneq ($(shell which python3),)
	PY = python3 # UTCS lab machines
else
	PY = python  # alternative
endif

CXX      = g++
CXXFLAGS = -std=c++14 -Ofast
LDFLAGS  = -L$(CUDADIR)
LDLIBS   = -lboost_program_options -lcuda -lcudart
NVFLAGS  = -std=c++11 -arch=sm_61 -O3
NVLDFLAGS= 
NVLDLIBS =

all: kmeans

.PHONY: add
add: format
	git add .

.PHONY: format
format: 
	clang-format -i *.c++ *.cu *.h *.cuh 

.PHONY: clean
clean:
	-rm -f *.o
	-rm -f *.a
	-rm -f *.gch
	-rm -f vgcore.*
	-rm -f kmeans

kmeans: kmeans.o km_cpu.o cuda.a
	$(CXX) $(LDFLAGS) kmeans.o km_cpu.o cuda.a -o kmeans $(LDLIBS)

kmeans.o: kmeans.c++ KMParams.h point.h km_cpu.h km_cuda.h
	$(CXX) $(CXXFLAGS) kmeans.c++ -c

point.o: point.cu point.h
	nvcc $(NVFLAGS) point.cu -dc

point.dlink.o: point.o
	nvcc $(NVFLAGS) point.o -dlink -o point.dlink.o
	nvcc $(NVFLAGS) point.o point.dlink.o -lib -o point.a

km_cuda.o: km_cuda.cu km_cuda.cuh km_cuda.h
	nvcc $(NVFLAGS) km_cuda.cu -dc

km_cuda.dlink.o: km_cuda.o point.o
	nvcc $(NVFLAGS) km_cuda.o point.o -dlink -o km_cuda.dlink.o

cuda.a: km_cuda.dlink.o point.dlink.o
	nvcc $(NVFLAGS) km_cuda.o km_cuda.dlink.o point.o point.dlink.o -lib -o cuda.a

km_cpu.o: km_cpu.c++ km_cpu.h KMParams.h point.h
	$(CXX) $(CXXFLAGS) km_cpu.c++ -c

results:
	mkdir results

.PHONY: tests
tests: kmeans results
	$(PY) scripts/tests.py results/step2-$(HOST).pdf
