#include <fstream>
#include <iostream>
#include <vector>
#include "KMParams.h"
using namespace std;

vector<double> read_input(string filename, int &n, int &dim);

int main(int argc, char const *argv[]) {
    KMParams kmp(argc, argv);
    kmp.print_params();

    int n, dim;
    vector<double> data = read_input(kmp.input, n, dim);
    cout << "input_metadata" << endl;
    cout << "n,dim" << endl;
    cout << n << endl;

    return 0;
}

/**
 * Reads a K-Means-lab-formatted input file into a new vector.
 * Used to read input for use with KMeans class.
 * @param[in]  filename path to the input file
 * @param[out] n        number of data points
 * @param[out] dim      dimension of each data point
 * @returns the data vector
 */
vector<double> read_input(string filename, int &n, int &dim) {
    vector<double> data;

    // open file
    ifstream stream;
    stream.open(filename);

    // read number of data points
    string _discard;
    stream >> n;
    getline(stream, _discard);

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
            data.push_back(val);
            dim_counter++;
        }
        dim = dim_counter;
    }

    return data;
}