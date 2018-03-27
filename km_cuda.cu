#include <chrono>
#include <iostream>
#include "KMParams.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "km_cuda.cuh"
#include "point.h"

__global__ void test_cuda(const KMParams *kmp, const point_data_t *data,
                          point_data_t *centroids) {
    for (int i = 0; i < kmp->clusters; ++i) {
        for (int j = 0; j < kmp->dim; ++j) {
            centroids[i * kmp->dim + j] = 777;
        }
    }
}

void km_cuda_run(const KMParams &host_kmp, const point_data_t *host_data,
                 point_data_t *host_centroids, std::chrono::duration<double> &t,
                 unsigned &iterations) {
    // copy params to device
    KMParams *kmp;
    cudaMalloc((void **)&kmp, sizeof(KMParams));
    cudaMemcpy(kmp, &host_kmp, sizeof(KMParams), cudaMemcpyHostToDevice);

    // copy points to device
    const unsigned data_size = host_kmp.n * host_kmp.dim * sizeof(point_data_t);
    point_data_t *data;
    cudaMalloc(&data, data_size);
    cudaMemcpy(data, host_data, data_size, cudaMemcpyHostToDevice);

    // copy centroids to device
    const unsigned centroids_size =
        host_kmp.clusters * host_kmp.dim * sizeof(point_data_t);
    point_data_t *centroids;
    cudaMalloc(&centroids, centroids_size);
    cudaMemcpy(centroids, host_centroids, centroids_size,
               cudaMemcpyHostToDevice);

    // run kernel
    test_cuda<<<1, 1>>>(kmp, data, centroids);
    // cudaDeviceSynchronize();

    std::cout << "Ran cuda kernel." << std::endl;

    // copy centroids back
    cudaMemcpy(host_centroids, centroids, centroids_size,
               cudaMemcpyDeviceToHost);

    cudaFree(centroids);
    cudaFree(data);
}
