#include "point.h"

CUDA_HOSTDEV
unsigned point_idx(unsigned point, unsigned dim, unsigned point_dim) {
    return point_dim * point + dim;
}

CUDA_HOSTDEV
unsigned point_idx(unsigned point, unsigned point_dim) {
    return point_idx(point_dim, point, 0);
}

CUDA_HOSTDEV
void point_copy(const point_data_t *data_src, unsigned idx_src,
                point_data_t *data_dst, unsigned idx_dst, unsigned point_dim) {
    for (int i = 0; i < point_dim; ++i) {
        data_dst[idx_dst + i] = data_src[idx_src + i];
    }
}

CUDA_HOSTDEV
point_data_t point_dist(const point_data_t *data_a, unsigned idx_a,
                        const point_data_t *data_b, unsigned idx_b,
                        unsigned point_dim) {
    point_data_t sum = 0;
    for (int i = 0; i < point_dim; ++i) {
        const point_data_t dx = data_b[idx_b + i] - data_a[idx_a + i];
        sum += dx * dx;
    }
    return sum;
}

CUDA_HOSTDEV
void point_add(const point_data_t *data_src, unsigned idx_src,
               point_data_t *data_dst, unsigned idx_dst, unsigned point_dim) {
    for (int i = 0; i < point_dim; ++i) {
        data_dst[idx_dst + i] += data_src[idx_src + i];
    }
}

CUDA_HOSTDEV
void point_scale(point_data_t *data_dst, unsigned idx_dst, point_data_t scalar,
                 unsigned point_dim) {
    for (int i = 0; i < point_dim; ++i) {
        data_dst[idx_dst + i] *= scalar;
    }
}
