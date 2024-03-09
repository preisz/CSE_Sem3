// g++ -std=c++17 -Wall -Werror -pedantic -pthread -fopenmp threads_omp.cpp
// ./a.out
#include <omp.h>

#include <iostream>
#include <cstddef>
#include <vector>

int main() {

  std::cout << omp_get_max_threads() << std::endl;

  auto values = std::vector<double>(10000);
#pragma omp for
  for (size_t n = 0; n < values.size(); ++n) {
    values[n] = 5;
  }

  return 0;
}