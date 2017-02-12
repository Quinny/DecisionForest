#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

/*
 * This file is pretty overkill, but i thought it was neat.
 */

namespace qp {
namespace rf {

// An implementation of the On combinator.  Returns F(Arg(x), Arg(y)).
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

// Gets the second item from a pair.
template <typename T, typename U>
struct Snd {
  const U& operator()(const std::pair<T, U>& p) const { return p.second; }
};

template <typename T, typename U, typename Cmp = std::less<U>>
using CompareOnSecond = On<std::pair<T, U>, Cmp, Snd<T, U>>;

}  // namespace rf
}  // namespace qp

#endif /* FUNCTIONAL_H */
