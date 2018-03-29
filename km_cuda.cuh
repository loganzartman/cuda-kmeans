#ifndef KM_CUDA_CUH
#define KM_CUDA_CUH

#include "cuda.h"
#include "km_cuda.h"
#include "point.h"
#define BLOCK_SIZE 256

#define cudachk(ans) \
    { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line,
                        bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Assert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}

__global__ void km_cuda_kernel(const KMParams *kmp, const point_data_t *data,
                               const point_data_t *centroids,
                               point_data_t *new_centroids,
                               unsigned *centroid_counts,
                               unsigned *centroid_map);

__device__ void km_cuda_map_nearest(const KMParams *kmp,
                                    const point_data_t *data,
                                    const point_data_t *centroids,
                                    unsigned *centroid_counts,
                                    unsigned *centroid_map);

__device__ void km_cuda_recompute_centroids(const KMParams *kmp,
                                            const point_data_t *data,
                                            point_data_t *new_centroids,
                                            const unsigned *centroid_counts,
                                            const unsigned *centroid_map);
#endif
