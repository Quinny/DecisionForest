#ifndef TREE_H
#define TREE_H

#include <algorithm>
#include <cmath>
#include "dataset.h"
#include "node.h"

namespace qp {
namespace rf {

// Enum denoting if the tree belongs to a single forest or a deep forest.
enum class TreeType { SINGLE_FOREST, DEEP_FOREST };

// A complete tree of DecisionNodes.
template <typename SplitterFn>
class DecisionTree {
 public:
  // Create a DecisionTree with a given depth.
  DecisionTree(int max_depth, TreeType type = TreeType::SINGLE_FOREST)
      : max_depth_(max_depth), type_(type), depth_(0) {}

  // Walks the tree based on the feature vector and returns the leaf node.
  const DecisionNode<SplitterFn>* walk(
      const std::vector<double>& features) const {
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
  double predict(const std::vector<double>& features) const {
    return walk(features)->predict();
  }

  // Train the tree on the given dataset.
  void train(SampledDataSet& data_set) {
    root_.reset(new DecisionNode<SplitterFn>());
    train_recurse(root_.get(), data_set.begin(), data_set.end(), 0);
  }

  // Eliminating the explicit recursion did not provide any speed ups.  The
  // depth is pretty shallow.
  void train_recurse(DecisionNode<SplitterFn>* current, SDIter first,
                     SDIter last, int current_depth) {
    // Record the current depth.
    depth_ = std::max(depth_, current_depth);

    // Train the current node.
    current->train(first, last);

    if (current->leaf() || current_depth == max_depth_) {
      current->make_leaf();
      return;
    }

    // Partition the dataset so that all LEFT examples are before all RIGHT
    // examples.
    auto pivot_iter = std::partition(first, last, [&](const auto& sample) {
      return current->split_direction(sample.get().features) ==
             SplitDirection::LEFT;
    });

    // Train the left and right nodes on the portion of the data that was split
    // to them.
    if (pivot_iter != first) {
      train_recurse(current->make_child(SplitDirection::LEFT), first,
                    pivot_iter, current_depth + 1);
    }

    if (pivot_iter != last) {
      train_recurse(current->make_child(SplitDirection::RIGHT), pivot_iter,
                    last, current_depth + 1);
    }
  }

  // Transform features based on a summation of activations.
  double transform_summation(const std::vector<double>& features) const {
    double sum = 0;
    const auto* current = root_.get();
    // Start at the root node and walk down the tree until we reach a leaf.
    while (!current->leaf()) {
      sum += current->activation(features);
      const auto dir = current->split_direction(features);
      const auto* next = current->get_child(dir);
      if (next == nullptr) break;
      current = next;
    }
    return sum;
  }

  int depth() const { return depth_; }

 private:
  std::unique_ptr<DecisionNode<SplitterFn>> root_;
  int max_depth_;
  int depth_;
  TreeType type_;
};

}  // namespace rf
}  // namespace qp

#endif /* TREE_H */
