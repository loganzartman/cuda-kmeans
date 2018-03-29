#ifndef KM_CUDA_CUH
#define KM_CUDA_CUH

#include "km_cuda.h"
#include "point.h"

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
