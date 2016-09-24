#include "criterion.h"
#include <cassert>
#include <iostream>

void zero_impurity() {
  // Only 1 label appears.
  std::map<int, std::size_t> label_histogram{{1, 10}};

  auto elements_impurity = qp::rf::gini_impurity(label_histogram);
  assert(elements_impurity.first == 10);
  assert(elements_impurity.second == 0);
}

void non_zero_impurity() {
  std::map<int, std::size_t> label_histogram{{1, 3}, {4, 7}, {5, 2}};

  // p(1) = 3/12 == .25, impurity = .1875
  // p(4) = 4/12 == .33, impurity = .2222
  // p(5) = 2/12 == .16, impurity = .1388
  // -------------------------------------
  //                                ~.5485
  auto elements_impurity = qp::rf::gini_impurity(label_histogram);
  assert(elements_impurity.first == 12);

  double expected = (3 / 12.0) * (1 - (3 / 12.0)) +
                    (4 / 12.0) * (1 - (4 / 12.0)) +
                    (2 / 12.0) * (1 - (2 / 12.0));
  std::cout << elements_impurity.second << " " << expected << std::endl;
}

int main() {
  zero_impurity();
  non_zero_impurity();
}
