#include <iostream>
#include <random>

#include "PAvx2.h"

int main() {
    /* minimal working example */

    int size = 10000000;
    auto *arr = new int32_t[size];

    std::random_device dev; // random number generator
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);

    for (uint32_t i = 0; i < size; i++) { // generate random array
        arr[i] = dist(rng);
    }

    PAvx2::sort(arr, size);
    std::cout << std::boolalpha << "sorted: " << std::is_sorted(arr, arr+size) << std::endl;

    for (uint32_t i = 0; i < size; i++) { // generate random array
        arr[i] = dist(rng);
    }

    PAvx2::parallel::sort(arr, size);
    std::cout << std::boolalpha << "sorted: " << std::is_sorted(arr, arr+size) << std::endl;

    return 0;
}
