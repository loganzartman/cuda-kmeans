#ifndef POINT_H
#define POINT_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

typedef double point_data_t;

CUDA_HOSTDEV
unsigned point_idx(unsigned point_dim, unsigned point);

CUDA_HOSTDEV
unsigned point_idx(unsigned point_dim, unsigned point, unsigned dim);

CUDA_HOSTDEV
void point_copy(const point_data_t *data_src, unsigned idx_src,
                point_data_t *data_dst, unsigned idx_dst, unsigned point_dim);

CUDA_HOSTDEV
point_data_t point_dist(const point_data_t *data_a, unsigned idx_a,
                        const point_data_t *data_b, unsigned idx_b,
                        unsigned point_dim);

CUDA_HOSTDEV
void point_add(const point_data_t *data_src, unsigned idx_src,
               point_data_t *data_dst, unsigned idx_dst, unsigned point_dim);

CUDA_HOSTDEV
void point_scale(point_data_t *data_dst, unsigned idx_dst, point_data_t scalar,
                 unsigned point_dim);

#endif
