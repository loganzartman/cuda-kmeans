CXX      = g++
CXXFLAGS = 
LDFLAGS  = 

all: kmeans

.PHONY: clean
clean:
	-rm -f *.o
	-rm -f kmeans

kmeans: kmeans.o
	$(CXX) $(LDFLAGS) kmeans.o -o kmeans

kmeans.o: kmeans.c++
	$(CXX) $(CXXFLAGS) kmeans.c++ -c