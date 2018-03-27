#include "km_cpu.h"
#include <algorithm>
#include "point.h"

void km_cpu_run(const KMParams &kmp, point_data_t *data,
                point_data_t *centroids) {
    using namespace std;

    // mapping of centroid index to number of associated points
    unsigned centroid_counts[kmp.clusters];

    // mapping of point index to associated centroid index
    unsigned centroid_map[kmp.n];

    // TODO: start timer

    // do work
    unsigned i = 0;
    while (i < kmp.iterations || kmp.iterations == 0) {
        km_cpu_map_nearest(kmp, data, centroids, centroid_counts, centroid_map);
        ++i;
    }

    // TODO: stop timer
}

void km_cpu_map_nearest(const KMParams &kmp, const point_data_t *data,
                        const point_data_t *centroids,
                        unsigned *centroid_counts, unsigned *centroid_map) {
    using namespace std;

    // find centroid nearest to each point
    for (int i = 0; i < kmp.n; ++i) {
        const int idx = i * kmp.dim;

        // find nearest
        int nearest = 0;
        point_data_t nearest_dist =
            point_dist(data, idx, centroids, 0, kmp.dim);

        for (int j = 0; j < kmp.clusters; ++j) {
            const int oidx = j * kmp.dim;
            point_data_t dist = point_dist(data, idx, centroids, oidx, kmp.dim);
            if (dist < nearest_dist) {
                nearest = j;
                nearest_dist = dist;
            }
        }

        // add to count
        ++centroid_counts[nearest];

        // add to mapping
        centroid_map[i] = nearest;
    }
}
