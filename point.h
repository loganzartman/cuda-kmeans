#ifndef POINT_H
#define POINT_H

typedef double point_data_t;

unsigned point_idx(unsigned point_dim, unsigned point);
unsigned point_idx(unsigned point_dim, unsigned point, unsigned dim);

void point_copy(const point_data_t *data_src, unsigned idx_src,
                point_data_t *data_dst, unsigned idx_dst, unsigned point_dim);

point_data_t point_dist(const point_data_t *data_a, unsigned idx_a,
                        const point_data_t *data_b, unsigned idx_b,
                        unsigned point_dim);

void point_add(const point_data_t *data_src, unsigned idx_src,
               point_data_t *data_dst, unsigned idx_dst, unsigned point_dim);

void point_scale(point_data_t *data_dst, unsigned idx_dst, unsigned point_dim,
                 point_data_t scalar);

#endif