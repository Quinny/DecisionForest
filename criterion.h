#ifndef CRITERION_H
#define CRITERION_H

#include <map>

namespace qp {
namespace rf {

// Given a histrogram of label occurances, compute the gini impurity of the
// distribution.  A return value of 0 means the distribution is totally pure
// (a single label), and a return value of 1 means the distribution is impure
// (all labels have an even number of occurances).
// Returns a pair of (total_elements, impurity).
// https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
template <typename Label>
std::pair<std::size_t, double> gini_impurity(
    const std::map<Label, std::size_t>& label_histogram) {
  std::size_t total_elements = 0;
  for (const auto& label_count : label_histogram) {
    total_elements += label_count.second;
  }

  double total_elements_real = static_cast<double>(total_elements);
  double impurity = 0;
  for (const auto& label_count : label_histogram) {
    double p = label_count.second / total_elements_real;
    impurity += p * (1 - p);
  }
  return {total_elements, impurity};
}

}  // namespace rf
}  // namespace qp

#endif /* CRITERION_H */
