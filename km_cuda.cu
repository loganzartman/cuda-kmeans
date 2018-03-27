#include <chrono>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "KMParams.h"
#include "point.h"
#include "km_cuda.cuh"

__global__
void test_cuda() {
    // no op
}

void km_cuda_run(const KMParams &kmp, const point_data_t *data,
                point_data_t *centroids, std::chrono::duration<double> &t,
                unsigned &iterations) {
    test_cuda<<<1, 1>>>();
    std::cout << "Ran cuda kernel." << std::endl;
}

