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
    bool cpu;
    bool print_points;
    bool shared_mem;

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
            "cpu", bool_switch()->default_value(false),
            "run CPU implementation instead of CUDA")(
            "print-points", bool_switch()->default_value(false),
            "output the list of input points in CSV format")(
            "no-shared-mem", bool_switch()->default_value(false),
            "disable CUDA shared memory");

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
        cpu = vm["cpu"].as<bool>();
        print_points = vm["print-points"].as<bool>();
        shared_mem = !vm["no-shared-mem"].as<bool>();
        n = 0;
        dim = 0;
    }
    KMParams(int clusters, double threshold, int workers, int iterations,
             bool cpu, bool print_points, bool shared_mem, unsigned n, unsigned dim)
        : clusters(clusters),
          threshold(threshold),
          workers(workers),
          iterations(iterations),
          cpu(cpu),
          print_points(print_points),
          shared_mem(shared_mem),
          n(n),
          dim(dim) {}
    KMParams(const KMParams &kmp) = default;
    ~KMParams() = default;

    void print_params() {
        using namespace std;
        cout << "params" << endl;
        cout << "clusters,threshold,workers,iterations,cpu,shared_mem" << endl;
        cout << clusters << ",";
        cout << threshold << ",";
        cout << workers << ",";
        cout << iterations << ",";
        cout << cpu << ",";
        cout << shared_mem << endl;
        cout << endl;
    }
};

#endif
