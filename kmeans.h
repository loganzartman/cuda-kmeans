#ifndef KMEANS_H
#define KMEANS_H

#include <string>
#include <vector>
#include "KMParams.h"
#include "point.h"

int main(int argc, char const *argv[]);
std::vector<point_data_t> read_input(KMParams &kmp);
void random_centroids(const KMParams &kmp, const point_data_t *data,
                      point_data_t *centroids);
void print_points(const KMParams &kmp, const point_data_t *data, unsigned n);

#endif