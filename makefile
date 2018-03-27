CXX      = g++
CXXFLAGS = -std=c++14
LDFLAGS  = 
LDLIBS   = -lboost_program_options

all: kmeans

.PHONY: add
add: format
	git add .

.PHONY: format
format: 
	clang-format -i *.c++ *.h

.PHONY: clean
clean:
	-rm -f *.o
	-rm -f *.gch
	-rm -f kmeans

kmeans: kmeans.o point.o km_cpu.o
	$(CXX) $(LDFLAGS) kmeans.o point.o km_cpu.o -o kmeans $(LDLIBS)

kmeans.o: kmeans.c++ KMParams.h point.h km_cpu.h
	$(CXX) $(CXXFLAGS) kmeans.c++ -c

point.o: point.c++ point.h
	$(CXX) $(CXXFLAGS) point.c++ -c

km_cpu.o: km_cpu.c++ km_cpu.h KMParams.h point.h
	$(CXX) $(CXXFLAGS) km_cpu.c++ -c