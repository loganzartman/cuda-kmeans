#include <chrono>
#include <iostream>
#include "KMParams.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "km_cuda.cuh"
#include "point.h"

void km_cuda_run(const KMParams &host_kmp, const point_data_t *host_data,
                 point_data_t *host_centroids, std::chrono::duration<double> &t,
                 unsigned &iterations) {
    using namespace std;

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

    // create mapping from centroid to number of points closest to it
    unsigned *host_centroid_counts =
        new unsigned[host_kmp.clusters * host_kmp.dim];
    unsigned *centroid_counts;
    cudaMalloc(&centroid_counts, host_kmp.clusters * host_kmp.dim);

    // create mapping from point to nearest centroid
    unsigned *host_centroid_map = new unsigned[host_kmp.n * host_kmp.dim];
    unsigned *centroid_map;
    cudaMalloc(&centroid_map, host_kmp.n * host_kmp.dim);

    // run kernel
    const int block_size = 256;
    const int num_blocks = (host_kmp.n + block_size - 1) / block_size;
    km_cuda_map_nearest<<<num_blocks, block_size>>>(
        kmp, data, centroids, centroid_counts, centroid_map);
    // cudaDeviceSynchronize();

    // copy counts back
    cudaMemcpy(host_centroid_counts, centroid_counts,
               host_kmp.clusters * host_kmp.dim, cudaMemcpyDeviceToHost);

    // copy mapping back
    cudaMemcpy(host_centroid_map, centroid_map, host_kmp.n * host_kmp.dim,
               cudaMemcpyDeviceToHost);

    // copy centroids back
    cudaMemcpy(host_centroids, centroids, centroids_size,
               cudaMemcpyDeviceToHost);

    for (int c = 0; c < host_kmp.clusters; ++c) {
        cout << host_centroid_counts[c] << " ";
    }
    cout << endl;

    delete[] host_centroid_counts;
    delete[] host_centroid_map;
    cudaFree(centroids);
    cudaFree(data);
}

__global__ void km_cuda_map_nearest(const KMParams *kmp,
                                    const point_data_t *data,
                                    const point_data_t *centroids,
                                    unsigned *centroid_counts,
                                    unsigned *centroid_map) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < kmp->n; i += stride) {
        const int idx = i * kmp->dim;

        // find nearest
        int nearest = 0;
        point_data_t nearest_dist =
            point_dist(data, idx, centroids, 0, kmp->dim);

        for (int j = 1; j < kmp->clusters; ++j) {
            const int oidx = j * kmp->dim;
            point_data_t dist =
                point_dist(data, idx, centroids, oidx, kmp->dim);
            if (dist < nearest_dist) {
                nearest = j;
                nearest_dist = dist;
            }
        }

        // add to count
        atomicAdd(&centroid_counts[nearest], 1);

        // add to mapping
        centroid_map[i] = nearest;
    }
}
