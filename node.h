#ifndef NODE_H
#define NODE_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "criterion.h"
#include "dataset.h"
#include "functional.h"

namespace qp {
namespace rf {

enum class SplitDirection { LEFT, RIGHT };

// Represents a single node in a decision tree.
template <typename SplitterFn>
class DecisionNode {
 public:
  DecisionNode() : leaf_(false){};

  // Train this node to decide on the dataset rows between start and end.
  void train(SDIter first, SDIter last) {
    // The prediction at this node is the most occurring label in the incoming
    // samples.
    prediction_ = mode_label(first, last);

    // If the dataset only contains one label, then make it a leaf decider
    // node.
    if (single_label(first, last)) {
      make_leaf();
    }

    double min_impurity = std::numeric_limits<double>::max();
    auto total_samples = static_cast<double>(last - first + 1);

    // Try different split functions and choose the one which results in the
    // least impurity.
    const auto total_features = first->get().features.size();
    const auto splits_to_try =
        std::sqrt(total_features) * splitter_.n_input_features();
    for (int i = 0; i < splits_to_try; ++i) {
      SplitterFn candidate_split;
      candidate_split.train(first, last);

      // Generate histograms for the number of instances from each class which
      // split left or right.
      LabelHistogram went_left, went_right;
      for (auto sample = first; sample != last; ++sample) {
        if (candidate_split.apply(sample->get().features) ==
            SplitDirection::LEFT) {
          ++went_left[sample->get().label];
        } else {
          ++went_right[sample->get().label];
        }
      }

      // Calculate the total impurity as a weighted average of the left and
      // right impurities.
      auto left_impurity = gini_impurity(went_left);
      auto right_impurity = gini_impurity(went_right);
      auto total_impurity =
          (left_impurity.first / total_samples) * left_impurity.second +
          (right_impurity.first / total_samples) * right_impurity.second;

      if (total_impurity < min_impurity) {
        min_impurity = total_impurity;
        splitter_ = std::move(candidate_split);
      }
    }
  }

  // Determine the direction of the split based on the features.
  SplitDirection split_direction(const std::vector<double>& features) const {
    return splitter_.apply(features);
  }

  // Predict the label at this node based on the mode label of the incoming
  // samples.
  double predict() const { return prediction_; }

  // Whether or not this node is ready to predict.
  bool leaf() const { return leaf_; }

  void make_leaf() {
    leaf_ = true;
    left_.reset();
    right_.reset();
  }

  // Allocate the child for the split direction and return a pointer to it.
  // If the child already exists, it will be overwritten.
  DecisionNode<SplitterFn>* make_child(SplitDirection dir) {
    if (dir == SplitDirection::LEFT) {
      left_.reset(new DecisionNode<SplitterFn>());
      return left_.get();
    } else {
      right_.reset(new DecisionNode<SplitterFn>());
      return right_.get();
    }
  }

  // Get the child at the split direction.  Will return nullptr if the child
  // has not been allocated.
  const DecisionNode<SplitterFn>* get_child(SplitDirection dir) const {
    return dir == SplitDirection::LEFT ? left_.get() : right_.get();
  }

  double activation(const std::vector<double>& features) const {
    return splitter_.activate(features);
  }

 private:
  std::unique_ptr<DecisionNode<SplitterFn>> left_, right_;

  double prediction_;
  SplitterFn splitter_;
  bool leaf_;
};

}  // namespace rf
}  // namespace qp

#endif /* NODE_H */
