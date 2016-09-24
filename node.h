#ifndef NODE_H
#define NODE_H

#include <algorithm>
#include <iostream>
#include <map>

#include "criterion.h"
#include "dataset.h"
#include "functional.h"

namespace qp {
namespace rf {

const int splits_to_try = 20;

enum class SplitDirection { LEFT, RIGHT };

// Represents a single node in a tree.
// TODO
//  - Premature decision making based on probability
//    - If this node was trained on 70% class 1, and 20% class 2, we are that
//      sure that example that makes it here belongs to either class.
// - Only decide on a subset of features.
template <typename Feature, typename Label, typename SplitterFn>
class DecisionNode {
 public:
  DecisionNode(){};

  void compute_probabilities(const SampledDataSet<Feature, Label>& dataset,
                             std::size_t start, std::size_t end) {
    for (auto i = start; i < end; ++i) {
      label_probabilities_[dataset[i].get().label] +=
          1 / static_cast<double>(end - start + 1);
    }
  }

  // Train this node to decide on the dataset rows between start and end.
  // TODO - Only on a subset of features too.
  void train(const SampledDataSet<Feature, Label>& dataset, std::size_t start,
             std::size_t end) {
    compute_probabilities(dataset, start, end);
    double min_impurity = 1000;
    auto total_samples = static_cast<double>(dataset.size());

    for (int i = 0; i < splits_to_try; ++i) {
      SplitterFn candidate_split;
      candidate_split.train(dataset, start, end);

      std::map<Label, std::size_t> went_left, went_right;
      for (unsigned sample = start; sample != end; ++sample) {
        if (candidate_split.apply(dataset[sample].get().features) ==
            SplitDirection::LEFT) {
          ++went_left[dataset[sample].get().label];
        } else {
          ++went_right[dataset[sample].get().label];
        }
      }

      auto left_impurity = gini_impurity(went_left);
      auto right_impurity = gini_impurity(went_right);
      auto total_impurity =
          (left_impurity.first / total_samples) * left_impurity.second +
          (right_impurity.first / total_samples) * right_impurity.second;

      if (total_impurity < min_impurity) {
        min_impurity = total_impurity;
        splitter_ = candidate_split;
      }
    }
  }

  // Name pending...
  SplitDirection split_direction(const std::vector<Feature>& features) const {
    return splitter_.apply(features);
  }

  // Predict the label at this node based on probabilities.
  Label predict() const {
    auto prediction_probability = std::max_element(
        label_probabilities_.begin(), label_probabilities_.end(),
        compare_on_second<Label, double>);
    return prediction_probability->first;
  }

 private:
  // bool leaf_node_;
  Label final_label_;
  SplitterFn splitter_;
  std::map<Label, double> label_probabilities_;
};

}  // namespace rf
}  // namespace qp

#endif /* NODE_H */
