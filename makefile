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

kmeans: kmeans.o
	$(CXX) $(LDFLAGS) kmeans.o -o kmeans $(LDLIBS)

kmeans.o: kmeans.c++ KMParams.h
	$(CXX) $(CXXFLAGS) kmeans.c++ KMParams.h -c