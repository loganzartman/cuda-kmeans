#include <iostream>
#include "KMParams.h"

using namespace std;
int main(int argc, char const *argv[]) {
    KMParams kmp(argc, argv);
    kmp.print_params();
    return 0;
}