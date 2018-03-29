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
    using namespace std::chrono;

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

    // create new centroids on device
    point_data_t *new_centroids;
    cudaMalloc(&new_centroids, centroids_size);

    // create old centroids on host
    point_data_t *host_old_centroids = new point_data_t[centroids_size];

    // create mapping from centroid to number of points closest to it
    unsigned *host_centroid_counts =
        new unsigned[host_kmp.clusters * host_kmp.dim];
    unsigned *centroid_counts;
    cudaMalloc(&centroid_counts, host_kmp.clusters * host_kmp.dim);

    // create mapping from point to nearest centroid
    unsigned *host_centroid_map = new unsigned[host_kmp.n * host_kmp.dim];
    unsigned *centroid_map;
    cudaMalloc(&centroid_map, host_kmp.n * host_kmp.dim);

    // start timer
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    const int block_size = 256;
    const int num_blocks = (host_kmp.n + block_size - 1) / block_size;

    unsigned i = 0;
    while (i < host_kmp.iterations || host_kmp.iterations == 0) {
        // clear output data
        cudaMemset(&new_centroids, 0, centroids_size);
        cudaMemset(&centroid_counts, 0, host_kmp.clusters * host_kmp.dim);

        // map nearest and sum new centroids
        km_cuda_kernel<<<num_blocks, block_size>>>(
            kmp, data, centroids, new_centroids, centroid_counts, centroid_map);

        // store old centroids
        for (int c = 0; c < host_kmp.clusters; ++c) {
            const int idx = c * host_kmp.dim;
            point_copy(host_centroids, idx, host_old_centroids, idx,
                       host_kmp.dim);
        }

        // copy centroids and counts back
        cudaMemcpy(host_centroids, new_centroids, centroids_size,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(host_centroid_counts, centroid_counts,
                   host_kmp.clusters * host_kmp.dim, cudaMemcpyDeviceToHost);

        // scale centroids to produce averages
        for (int c = 0; c < host_kmp.clusters; ++c) {
            const int idx = c * host_kmp.dim;
            const point_data_t scalar =
                (point_data_t)1 / host_centroid_counts[c];
            point_scale(host_centroids, idx, scalar, host_kmp.dim);
        }

        ++i;

        // test convergence
        if (km_cuda_converged(host_kmp, host_old_centroids, host_centroids))
            break;

        // store new centroids into old
        cudaMemcpy(centroids, host_centroids, centroids_size,
                   cudaMemcpyHostToDevice);
    }
    iterations = i;

    // stop timer
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    t = t2 - t1;

    delete[] host_old_centroids;
    delete[] host_centroid_counts;
    delete[] host_centroid_map;
    cudaFree(centroids);
    cudaFree(new_centroids);
    cudaFree(data);
}

bool km_cuda_converged(const KMParams &kmp, const point_data_t *old_centroids,
                       const point_data_t *centroids) {
    using namespace std;

    // compute maximum distance from old centroid to new centroid
    point_data_t maxdist = 0;
    for (int i = 0; i < kmp.clusters; ++i) {
        const unsigned idx = i * kmp.dim;
        const point_data_t dist =
            point_dist(old_centroids, idx, centroids, idx, kmp.dim);
        maxdist = dist > maxdist ? dist : maxdist;
    }
    return maxdist < kmp.threshold;
}

__global__ void km_cuda_kernel(const KMParams *kmp, const point_data_t *data,
                               const point_data_t *centroids,
                               point_data_t *new_centroids,
                               unsigned *centroid_counts,
                               unsigned *centroid_map) {
    km_cuda_map_nearest(kmp, data, centroids, centroid_counts, centroid_map);
    km_cuda_recompute_centroids(kmp, data, new_centroids, centroid_counts,
                                centroid_map);
}

__device__ void km_cuda_map_nearest(const KMParams *kmp,
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

__device__ void km_cuda_recompute_centroids(const KMParams *kmp,
                                            const point_data_t *data,
                                            point_data_t *new_centroids,
                                            const unsigned *centroid_counts,
                                            const unsigned *centroid_map) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // sum
    for (int i = index; i < kmp->n; i += stride) {
        const int idx = i * kmp->dim;
        const int cidx = centroid_map[i] * kmp->dim;
        for (int d = 0; d < kmp->dim; ++d)
            atomicAdd(&new_centroids[cidx + d], data[idx + d]);
    }
}
