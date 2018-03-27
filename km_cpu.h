#ifndef KM_CPU_H
#define KM_CPU_H

#include <chrono>
#include "KMParams.h"
#include "point.h"

void km_cpu_run(const KMParams &kmp, const point_data_t *data,
                point_data_t *centroids, std::chrono::duration<double> &t,
                unsigned &iterations);

void km_cpu_map_nearest(const KMParams &kmp, const point_data_t *data,
                        const point_data_t *centroids,
                        unsigned *centroid_counts, unsigned *centroid_map);

void km_cpu_recompute_centroids(const KMParams &kmp, const point_data_t *data,
                                point_data_t *centroids,
                                const unsigned *centroid_counts,
                                const unsigned *centroid_map);

bool km_cpu_converged(const KMParams &kmp, const point_data_t *old_centroids,
                      const point_data_t *centroids);

#endif