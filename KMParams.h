#ifndef KMPARAMS_H
#define KMPARAMS_H

#include <boost/program_options.hpp>
#include <iostream>
#include <string>

struct KMParams {
    int clusters;
    double threshold;
    int workers;
    int iterations;
    unsigned n;
    unsigned dim;
    std::string input;

    KMParams(int argc, const char *argv[]) {
        using namespace std;
        using namespace boost::program_options;
        variables_map vm;

        // define options
        options_description desc("Logan's K-Means Solver\nAllowed Options");
        desc.add_options()("help", "shows usage information")(
            "clusters", value<int>()->default_value(1),
            "the number of clusters to find")(
            "threshold", value<double>()->default_value(0.0000001f),
            "the threshold for convergence")(
            "iterations", value<int>()->default_value(0),
            "maximum number of k-means iterations")(
            "workers", value<int>()->default_value(1), "number of threads")(
            "input", value<string>()->required(), "input file path")(
            "output-points", bool_switch()->default_value(false),
            "output input points");

        // parse options
        store(parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
            cout << desc << endl;
        notify(vm);

        // extract options
        clusters = vm["clusters"].as<int>();
        threshold = vm["threshold"].as<double>();
        iterations = vm["iterations"].as<int>();
        workers = vm["workers"].as<int>();
        input = vm["input"].as<string>();
        n = -1;
        dim = -1;
    }
    KMParams(int clusters, double threshold, int workers, int iterations,
             unsigned n, unsigned dim)
        : clusters(clusters),
          threshold(threshold),
          workers(workers),
          iterations(iterations),
          n(n),
          dim(dim) {}
    KMParams(const KMParams &kmp) = default;
    ~KMParams() = default;

    void print_params() {
        using namespace std;
        cout << "params" << endl;
        cout << "clusters,threshold,workers,iterations" << endl;
        cout << clusters << ",";
        cout << threshold << ",";
        cout << workers << ",";
        cout << iterations << endl;
        cout << endl;
    }
};

#endif