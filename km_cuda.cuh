#ifndef KM_CUDA_CUH
#define KM_CUDA_CUH

#include "km_cuda.h"
#include "point.h"

__global__ void km_cuda_map_nearest(const KMParams *kmp,
                                    const point_data_t *data,
                                    const point_data_t *centroids,
                                    unsigned *centroid_counts,
                                    unsigned *centroid_map);

#endif
