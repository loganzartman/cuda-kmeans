#ifndef KM_CUDA_H
#define KM_CUDA_H

#include <chrono>
#include "KMParams.h"
#include "point.h"

void km_cuda_run(const KMParams &kmp, const point_data_t *data,
                point_data_t *centroids, std::chrono::duration<double> &t,
                unsigned &iterations);

#endif
