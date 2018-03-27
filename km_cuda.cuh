#ifndef KM_CUDA_CUH
#define KM_CUDA_CUH

#include "km_cuda.h"

__global__ void test_cuda(const KMParams &kmp, const point_data_t *data,
                          point_data_t *centroids);

#endif
