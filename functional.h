#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace qp {
namespace rf {

template <typename Arg, typename Comb, typename F>
class On {
 public:
  using return_type = typename std::result_of<Comb(
      typename std::result_of<F(Arg)>::type,
      typename std::result_of<F(Arg)>::type)>::type;

  return_type operator()(const Arg& arg1, const Arg& arg2) const {
    return c(f(arg1), f(arg2));
  }

 private:
  F f;
  Comb c;
};

template <typename T, typename U>
struct Snd {
  const U& operator()(const std::pair<T, U>& p) const { return p.second; }
};

template <typename T, typename U, typename Cmp = std::less<U>>
using CompareOnSecond = On<std::pair<T, U>, Cmp, Snd<T, U>>;

template <typename Container, typename Generator>
void generate_back_n(Container& c, std::size_t n, const Generator& g) {
  std::generate_n(std::back_inserter(c), n, g);
}

template <typename T>
std::vector<T> project(const std::vector<T>& v,
                       const std::vector<std::size_t>& indicies) {
  std::vector<T> ret;
  ret.reserve(indicies.size());
  for (const auto& i : indicies) {
    ret.push_back(v[i]);
  }
  return ret;
}

}  // namespace rf
}  // namespace qp

#endif /* FUNCTIONAL_H */
