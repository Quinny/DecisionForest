#ifndef CRITERION_H
#define CRITERION_H

#include <map>

namespace qp {
namespace rf {

template <typename Label>
std::pair<std::size_t, double> gini_impurity(
    const std::map<Label, std::size_t>& label_histogram) {
  std::size_t total_elements = 0;
  for (const auto& label_count : label_histogram) {
    total_elements += label_count.second;
  }

  double impurity = 0;
  for (const auto& label_count : label_histogram) {
    double p = label_count.second / static_cast<double>(total_elements);
    impurity += p * (1 - p);
  }
  return {total_elements, impurity};
}

}  // namespace rf
}  // namespace qp

#endif /* CRITERION_H */
