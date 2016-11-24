#ifndef VECTOR_UTIL_H
#define VECTOR_UTIL_H

#include <algorithm>
#include <vector>

namespace qp {
namespace rf {

// Add an element to each entry in a vector.
void vector_plus(std::vector<double>& vec, const double v) {
  for (auto& e : vec) e += v;
}

// Subtract an element from each entry in a vector.
void vector_minus(std::vector<double>& vec, const double v) {
  for (auto& e : vec) e -= v;
}

// Element wise minus of dst[i] - src[i].
void vector_minus(std::vector<double>& dst, const std::vector<double>& src) {
  for (std::size_t i = 0; i < std::min(dst.size(), src.size()); ++i) {
    dst[i] -= src[i];
  }
}

// Element wise plus of dst[i] - src[i].
void vector_plus(std::vector<double>& dst, const std::vector<double>& src) {
  for (std::size_t i = 0; i < std::min(dst.size(), src.size()); ++i) {
    dst[i] += src[i];
  }
}

template <typename Container, typename Generator>
void generate_back_n(Container& c, std::size_t n, const Generator& g) {
  std::generate_n(std::back_inserter(c), n, g);
}

template <typename T, typename Iter>
void project(const std::vector<T>& v, const std::vector<std::size_t>& indices,
             Iter out) {
  std::transform(indices.begin(), indices.end(), out,
                 [&v](const std::size_t i) { return v[i]; });
}

template <typename T>
std::vector<T> project(const std::vector<T>& v,
                       const std::vector<std::size_t>& indicies) {
  std::vector<T> ret(indicies.size());
  project(v, indicies, ret.begin());
  return ret;
}

}  // namespace rf
}  // namespace qp

#endif /* VECTOR_UTIL_H */
