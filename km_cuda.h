#ifndef KM_CUDA_H
#define KM_CUDA_H

#include <chrono>
#include "KMParams.h"
#include "point.h"

void km_cuda_run(const KMParams &host_kmp, const point_data_t *host_data,
                 point_data_t *host_centroids, std::chrono::duration<double> &t,
                 unsigned &iterations);

bool km_cuda_converged(const KMParams &kmp, const point_data_t *old_centroids,
                       const point_data_t *centroids);

#endif
