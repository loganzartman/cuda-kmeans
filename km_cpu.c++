#include "km_cpu.h"
#include <algorithm>
#include <chrono>
#include "point.h"

void km_cpu_run(const KMParams &kmp, point_data_t *data,
                point_data_t *centroids, std::chrono::duration<double> &t) {
    using namespace std;
    using namespace std::chrono;

    // mapping of centroid index to number of associated points
    unsigned centroid_counts[kmp.clusters];

    // mapping of point index to associated centroid index
    unsigned centroid_map[kmp.n];

    // start timer
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // do work
    unsigned i = 0;
    while (i < kmp.iterations || kmp.iterations == 0) {
        // reset counts
        fill(centroid_counts, centroid_counts + kmp.clusters, 0);

        // find nearest centroids to each point
        km_cpu_map_nearest(kmp, data, centroids, centroid_counts, centroid_map);

        // compute new centroids as avg of all points mapped to each centroid
        km_cpu_recompute_centroids(kmp, data, centroids, centroid_counts,
                                   centroid_map);

        // TODO: convergence test

        ++i;
    }

    // stop timer
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    t = t2 - t1;
}

/**
 * Maps each centroid to the set of points nearer to it than another centroid.
 * @param[in]  kmp             the K-Means parameters
 * @param[in]  data            the array of point data
 * @param[in]  centroids       the array of centroid data
 * @param[out] centroid_counts mapping centroid index to # associated points
 * @param[out] centroid_map    mapping point index to associated centroid
 */
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

void km_cpu_recompute_centroids(const KMParams &kmp, const point_data_t *data,
                                point_data_t *centroids,
                                const unsigned *centroid_counts,
                                const unsigned *centroid_map) {
    using namespace std;

    // sum
    for (int i = 0; i < kmp.n; ++i) {
        const int idx = i * kmp.dim;
        const int cidx = centroid_map[i] * kmp.dim;
        point_add(data, idx, centroids, cidx, kmp.dim);
    }

    // average
    for (int c = 0; c < kmp.clusters; ++c) {
        const int cidx = centroid_map[c] * kmp.dim;
        const point_data_t scalar = (point_data_t)1 / centroid_counts[c];
        point_scale(centroids, cidx, scalar, kmp.dim);
    }
}