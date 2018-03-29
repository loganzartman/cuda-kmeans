#include <chrono>
#include <iostream>
#include "KMParams.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "km_cuda.cuh"
#include "point.h"

__device__ bool converged;

void km_cuda_run(const KMParams &host_kmp, const point_data_t *host_data,
                 point_data_t *host_centroids, std::chrono::duration<double> &t,
                 unsigned &iterations) {
    using namespace std;
    using namespace std::chrono;

    // copy params to device
    KMParams *kmp;
    cudachk(cudaMalloc((void **)&kmp, sizeof(KMParams)));
    cudachk(
        cudaMemcpy(kmp, &host_kmp, sizeof(KMParams), cudaMemcpyHostToDevice));

    // copy points to device
    const unsigned data_size = host_kmp.n * host_kmp.dim * sizeof(point_data_t);
    point_data_t *data;
    cudachk(cudaMalloc(&data, data_size));
    cudachk(cudaMemcpy(data, host_data, data_size, cudaMemcpyHostToDevice));

    // copy centroids to device
    const unsigned centroids_size =
        host_kmp.clusters * host_kmp.dim * sizeof(point_data_t);
    point_data_t *centroids;
    cudachk(cudaMalloc(&centroids, centroids_size));
    cudachk(cudaMemcpy(centroids, host_centroids, centroids_size,
                       cudaMemcpyHostToDevice));

    // create new centroids on device
    point_data_t *new_centroids;
    cudachk(cudaMalloc(&new_centroids, centroids_size));

    // create mapping from centroid to number of points closest to it
    const unsigned centroid_counts_size =
        host_kmp.clusters * host_kmp.dim * sizeof(unsigned);
    unsigned *host_centroid_counts =
        new unsigned[host_kmp.clusters * host_kmp.dim];
    unsigned *centroid_counts;
    cudachk(cudaMalloc(&centroid_counts, centroid_counts_size));

    // create mapping from point to nearest centroid
    const unsigned centroid_map_size =
        host_kmp.n * host_kmp.dim * sizeof(unsigned);
    unsigned *host_centroid_map = new unsigned[host_kmp.n * host_kmp.dim];
    unsigned *centroid_map;
    cudachk(cudaMalloc(&centroid_map, centroid_map_size));

    // store converged status
    bool host_converged = false;
    cudachk(cudaMemcpyToSymbol(converged, &host_converged, sizeof(bool)));

    // start timer
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    const int num_blocks = (host_kmp.n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unsigned i = 0;
    while (i < host_kmp.iterations || host_kmp.iterations == 0) {
        // clear output data
        cudachk(cudaMemset(new_centroids, 0, centroids_size));
        cudachk(cudaMemset(centroid_counts, 0, centroid_counts_size));

        if (SHARED_MEM_FLAG) {
            // find nearest centroids
            km_cuda_map_nearest<<<num_blocks, BLOCK_SIZE,centroid_counts_size>>>(
                kmp, data, centroids, centroid_counts, centroid_map);

            // compute new centroids
            km_cuda_recompute_centroids<<<num_blocks, BLOCK_SIZE, centroids_size>>>(
                kmp, data, new_centroids, centroid_counts, centroid_map);
        } else {
            km_cuda_map_nearest_noshare<<<num_blocks, BLOCK_SIZE>>>(
                kmp, data, centroids, centroid_counts, centroid_map);

            km_cuda_recompute_centroids_noshare<<<num_blocks, BLOCK_SIZE>>>(
                kmp, data, new_centroids, centroid_counts, centroid_map);
        }

        // test convergence
        km_cuda_convergence<<<num_blocks, BLOCK_SIZE>>>(kmp, centroids,
                                                        new_centroids);

        ++i;

        // check convergence
        cudachk(cudaMemcpyFromSymbol(&host_converged, converged, sizeof(bool)));
        if (host_converged)
            break;

        // store new centroids into old
        cudachk(cudaMemcpy(centroids, new_centroids, centroids_size,
                           cudaMemcpyDeviceToDevice));
    }
    iterations = i;

    // copy centroids back
    cudachk(cudaMemcpy(host_centroids, new_centroids, centroids_size,
                       cudaMemcpyDeviceToHost));

    // stop timer
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    t = t2 - t1;

    delete[] host_centroid_counts;
    delete[] host_centroid_map;
    cudaFree(centroids);
    cudaFree(new_centroids);
    cudaFree(data);
}

__global__ void km_cuda_map_nearest(const KMParams *kmp,
                                    const point_data_t *data,
                                    const point_data_t *centroids,
                                    unsigned *centroid_counts,
                                    unsigned *centroid_map) {
    extern __shared__ point_data_t shared_counts[];

    // clear shared counts
    for (int i = threadIdx.x; i < kmp->clusters; i += blockDim.x)
        shared_counts[i] = 0;
    __syncthreads();

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
        atomicAdd(&shared_counts[nearest], 1);

        // add to mapping
        centroid_map[i] = nearest;
    }
    __syncthreads();

    // aggregate
    for (int c = threadIdx.x; c < kmp->clusters; c += blockDim.x)
        atomicAdd(&centroid_counts[c], shared_counts[c]);
}

__global__ void km_cuda_map_nearest_noshare(const KMParams *kmp,
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

__global__ void km_cuda_recompute_centroids(const KMParams *kmp,
                                            const point_data_t *data,
                                            point_data_t *new_centroids,
                                            const unsigned *centroid_counts,
                                            const unsigned *centroid_map) {
    extern __shared__ point_data_t shared_centroids[];

    // clear shared centroids
    for (int i = threadIdx.x; i < kmp->clusters * kmp->dim; i += blockDim.x)
        shared_centroids[i] = 0;
    __syncthreads();

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // sum
    for (int i = index; i < kmp->n; i += stride) {
        const int c = centroid_map[i];
        const int idx = i * kmp->dim;
        const int cidx = c * kmp->dim;
        for (int d = 0; d < kmp->dim; ++d)
            atomicAdd(&shared_centroids[cidx + d], data[idx + d]);
    }
    __syncthreads();

    for (int c = threadIdx.x; c < kmp->clusters; c += blockDim.x) {
        const int cidx = c * kmp->dim;
        const point_data_t scalar = (point_data_t)1 / centroid_counts[c];
        for (int d = 0; d < kmp->dim; ++d)
            atomicAdd(&new_centroids[cidx + d], shared_centroids[cidx + d] * scalar);
    }
}

__global__ void km_cuda_recompute_centroids_noshare(
    const KMParams *kmp, const point_data_t *data, point_data_t *new_centroids,
    const unsigned *centroid_counts, const unsigned *centroid_map) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // sum
    for (int i = index; i < kmp->n; i += stride) {
        const int c = centroid_map[i];
        const int idx = i * kmp->dim;
        const int cidx = c * kmp->dim;
        const point_data_t scalar = (point_data_t)1 / centroid_counts[c];
        for (int d = 0; d < kmp->dim; ++d)
            atomicAdd(&new_centroids[cidx + d], data[idx + d] * scalar);
    }
}

__global__ void km_cuda_convergence(const KMParams *kmp,
                                    const point_data_t *old_centroids,
                                    const point_data_t *centroids) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    // compute maximum distance from old centroid to new centroid
    point_data_t maxdist = 0;
    for (int i = 0; i < kmp->clusters; ++i) {
        const unsigned idx = i * kmp->dim;
        const point_data_t dist =
            point_dist(old_centroids, idx, centroids, idx, kmp->dim);
        maxdist = dist > maxdist ? dist : maxdist;
    }
    converged = maxdist < kmp->threshold;
}