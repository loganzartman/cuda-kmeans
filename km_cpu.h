#ifndef KM_CPU_H
#define KM_CPU_H

#include "KMParams.h"
#include "point.h"

void km_cpu_run(const KMParams &kmp, point_data_t *data,
                point_data_t *centroids);

#endif