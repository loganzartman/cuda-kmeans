#include <fstream>
#include <iostream>
#include "KMParams.h"
using namespace std;

double *read_input(string filename, int &n, int &dim);

int main(int argc, char const *argv[]) {
    KMParams kmp(argc, argv);
    kmp.print_params();

    int n, dim;
    double *data = read_input(kmp.input, n, dim);
    cout << "input_metadata" << endl;
    cout << "n,dim" << endl;
    cout << n << endl;

    delete[] data;
    return 0;
}

/**
 * Reads a K-Means-lab-formatted input file into a new array.
 * Used to read input for use with KMeans class.
 * @param[in]  filename path to the input file
 * @param[out] n        number of data points
 * @param[out] dim      dimension of each data point
 * @returns a pointer to the heap-allocated data array
 */
double *read_input(string filename, int &n, int &dim) {
    // open file
    ifstream stream;
    stream.open(filename);

    // read number of data points
    string _discard;
    stream >> n;
    getline(stream, _discard);

    double *data = new double[n];

    // read each data point
    for (int i = 0; i < n; i++) {
        int index;
        stream >> index;

        string line;
        getline(stream, line);
        istringstream line_stream(line);

        int dim_counter = 0;
        double val;
        while (line_stream >> val) {
            data[i] = val;
            dim_counter++;
        }
        dim = dim_counter;
    }

    return data;
}