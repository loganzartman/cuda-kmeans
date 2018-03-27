#include "kmeans.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "KMParams.h"
#include "km_cpu.h"
#include "km_cuda.h"
#include "point.h"
using namespace std;

int main(int argc, char const *argv[]) {
    using namespace std::chrono;

    KMParams kmp(argc, argv);
    kmp.print_params();

    vector<point_data_t> data_vec = read_input(kmp);
    cout << "input_metadata" << endl;
    cout << "n,dim" << endl;
    cout << kmp.n << "," << kmp.dim << endl;
    cout << endl;

    // copy data into array
    point_data_t *data = new point_data_t[kmp.n * kmp.dim];
    copy(data_vec.begin(), data_vec.end(), data);

    // randomize centroids
    point_data_t *centroids = new point_data_t[kmp.clusters * kmp.dim];
    random_centroids(kmp, data, centroids);

    // run k-means
    unsigned iterations;
    duration<double> dt;
    if (kmp.cpu)
        km_cpu_run(kmp, data, centroids, dt, iterations);
    else
        km_cuda_run(kmp, data, centroids, dt, iterations);

    // output timing
    cout << "statistics" << endl;
    cout << "time_us,iterations" << endl;
    cout << duration_cast<microseconds>(dt).count();
    cout << "," << iterations << endl << endl;

    // output centroids
    cout << "centroids" << endl;
    print_points(kmp, centroids, kmp.clusters);
    cout << endl;

    delete[] data;
    delete[] centroids;
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
vector<point_data_t> read_input(KMParams &kmp) {
    vector<point_data_t> data;

    // open file
    ifstream stream;
    stream.open(kmp.input);

    // read number of data points
    string _discard;
    stream >> kmp.n;
    getline(stream, _discard);

    // read each data point
    for (int i = 0; i < kmp.n; i++) {
        int index;
        stream >> index;

        string line;
        getline(stream, line);
        istringstream line_stream(line);

        unsigned dim_counter = 0;
        double val;
        while (line_stream >> val) {
            data.push_back(val);
            dim_counter++;
        }
        kmp.dim = dim_counter;
    }

    return data;
}

/**
 * Populates a point array with randomized centroids.
 * @param[in]  kmp       K-Means parameters
 * @param[in]  data      point data to select from
 * @param[out] centroids array to be populated
 */
void random_centroids(const KMParams &kmp, const point_data_t *data,
                      point_data_t *centroids) {
    using namespace std;

    // create random machinery
    random_device rd;
    mt19937 twister(rd());

    // create indices
    unsigned *indices = new unsigned[kmp.n];
    iota(indices, indices + kmp.n, 0);
    shuffle(indices, indices + kmp.n, twister);

    for (int c = 0; c < kmp.clusters; ++c) {
        const unsigned src = indices[c];
        point_copy(data, src * kmp.dim, centroids, c * kmp.dim, kmp.dim);
    }
    delete[] indices;
}

/**
 * Prints an array of points in CSV format.
 * @param[in] kmp  K-Means parameters
 * @param[in] data data to print
 * @param[in] n    how many points are in data
 */
void print_points(const KMParams &kmp, const point_data_t *data, unsigned n) {
    // print header
    for (int d = 0; d < kmp.dim; ++d) {
        if (d > 0)
            cout << ",";
        cout << "dim_" << d;
    }
    cout << endl;

    // print data
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < kmp.dim; ++d) {
            if (d > 0)
                cout << ",";
            cout << data[i * kmp.dim + d];
        }
        cout << endl;
    }
}
