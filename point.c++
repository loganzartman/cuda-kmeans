#include "point.h"

unsigned point_idx(unsigned point, unsigned dim, unsigned point_dim) {
    return point_dim * point + dim;
}

unsigned point_idx(unsigned point, unsigned point_dim) {
    return point_idx(point_dim, point, 0);
}

void point_copy(const point_data_t *data_src, unsigned idx_src,
                point_data_t *data_dst, unsigned idx_dst, unsigned point_dim) {
    for (int i = 0; i < point_dim; ++i) {
        data_dst[idx_dst + i] = data_src[idx_src + i];
    }
}

void point_add(const point_data_t *data_src, unsigned idx_src,
               point_data_t *data_dst, unsigned idx_dst, unsigned point_dim) {
    for (int i = 0; i < point_dim; ++i) {
        data_dst[idx_dst + i] += data_src[idx_src + i];
    }
}

void point_scale(point_data_t *data_dst, unsigned idx_dst, unsigned point_dim,
                 point_data_t scalar) {
    for (int i = 0; i < point_dim; ++i) {
        data_dst[idx_dst + i] *= scalar;
    }
}
