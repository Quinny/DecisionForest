#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <functional>
#include <utility>

namespace qp {
namespace rf {

template <typename T, typename U, typename Cmp = std::less<U>>
bool compare_on_second(const std::pair<T, U>& lhs, const std::pair<T, U>& rhs) {
  Cmp c;
  return c(lhs.second, rhs.second);
}

}  // namespace rf
}  // namespace qp

#endif /* FUNCTIONAL_H */
