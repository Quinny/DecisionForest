#ifndef TREE_H
#define TREE_H

#include <algorithm>
#include <cmath>
#include "dataset.h"
#include "node.h"

namespace qp {
namespace rf {

// A complete tree of DecisionNodes.
template <typename Feature, typename Label, typename SplitterFn>
class DecisionTree {
 public:
  // Create a DecisionTree with a given depth.
  DecisionTree(int max_depth) : max_depth_(max_depth) {}

  // Walks the tree based on the feature vector and returns the leaf node.
  const DecisionNode<Feature, Label, SplitterFn>* walk(
      const std::vector<Feature>& features) const {
    const auto* current = root_.get();
    // Start at the root node and walk down the tree until we reach a leaf.
    while (!current->leaf()) {
      const auto dir = current->split_direction(features);
      const auto* next = current->get_child(dir);
      if (next == nullptr) break;
      current = next;
    }
    return current;
  }

  // Predict the label for a set of features.
  Label predict(const std::vector<Feature>& features) const {
    return walk(features)->predict();
  }

  // Train the tree on the given dataset.
  void train(SampledDataSet<Feature, Label>& data_set) {
    root_.reset(new DecisionNode<Feature, Label, SplitterFn>());
    train_recurse(data_set, root_.get(), /* start=*/0,
                  /* end=*/data_set.size(), 0);
  }

  // Eliminating the explicit recursion did not provide any speed ups.  The
  // depth is pretty shallow.
  void train_recurse(SampledDataSet<Feature, Label>& data_set,
                     DecisionNode<Feature, Label, SplitterFn>* current,
                     std::size_t start, std::size_t end, int current_depth) {
    // Train the current node.
    current->train(data_set, start, end);

    // The the node is a leaf then initialize the mahalanobis distance
    // calculator for feature transformation.
    if (current->leaf() || current_depth == max_depth_) {
      current->make_leaf();
      current->initialize_mahalanobis(data_set, start, end);
      return;
    }

    // Partition the dataset so that all LEFT examples are before all RIGHT
    // examples.
    auto pivot_iter = std::partition(
        data_set.begin() + start, data_set.begin() + end,
        [&](const auto& sample) {
          return current->split_direction(sample.get().features) ==
                 SplitDirection::LEFT;
        });

    auto mid = std::distance(data_set.begin(), pivot_iter);

    // Train the left and right nodes on the portion of the data that was split
    // to them.
    if (mid != start) {
      train_recurse(data_set, current->make_child(SplitDirection::LEFT), start,
                    mid, current_depth + 1);
    }

    if (mid != end) {
      train_recurse(data_set, current->make_child(SplitDirection::RIGHT), mid,
                    end, current_depth + 1);
    }
  }

 private:
  std::unique_ptr<DecisionNode<Feature, Label, SplitterFn>> root_;
  int max_depth_;
};

}  // namespace rf
}  // namespace qp

#endif /* TREE_H */
