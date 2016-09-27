#ifndef RANDOM_H
#define RANDOM_H

#include <ctime>
#include <random>

namespace qp {
namespace rf {

// Generates a random integral in the range [first, last].
template <typename T>
T random_range(const T& first, const T& last) {
  static std::knuth_b gen(std::time(nullptr));
  std::uniform_int_distribution<T> dist(first, last);
  return dist(gen);
}

}  // namespace rf
}  // namespace qp

#endif /* RANDOM_H */
