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
  DecisionTree(std::size_t depth) : nodes_(std::pow(2, depth + 1) - 1) {}

  // Checks if a given node index is a leaf.
  bool is_leaf(std::size_t i) const { return 2 * i + 1 >= nodes_.size(); }

  // Given a direction and a node index, return the index of the corresponding
  // child.
  std::size_t walk_to(SplitDirection dir, std::size_t i) const {
    return dir == SplitDirection::LEFT ? 2 * i + 1 : 2 * i + 2;
  }

  // Predict the label for a set of features.
  Label predict(const std::vector<Feature>& features) const {
    std::size_t node_index = 0;

    // Start at the root node and walk down the tree until we reach a leaf.
    while (!is_leaf(node_index)) {
      node_index =
          walk_to(nodes_[node_index].split_direction(features), node_index);
    }
    return nodes_[node_index].predict();
  }

  // Train the tree on the given dataset.
  void train(SampledDataSet<Feature, Label>& data_set) {
    train_recurse(data_set, /* node_index=*/0, /* start=*/0,
                  /* end=*/data_set.size());
  }

  // Eliminating the explicit recursion did not provide any speed ups.  The
  // depth is pretty shallow.
  void train_recurse(SampledDataSet<Feature, Label>& data_set,
                     std::size_t node_index, std::size_t start,
                     std::size_t end) {
    if (node_index >= nodes_.size() || start >= end) {
      return;
    }

    // Train the current node.
    nodes_[node_index].train(data_set, start, end);

    // Partition the dataset so that all LEFT examples are before all RIGHT
    // examples.
    auto pivot_iter = std::partition(
        data_set.begin() + start, data_set.begin() + end,
        [&](const auto& sample) {
          return nodes_[node_index].split_direction(sample.get().features) ==
                 SplitDirection::LEFT;
        });

    auto mid = std::distance(data_set.begin(), pivot_iter);

    // Train the left and right nodes on the portion of the data that was split
    // to them.
    train_recurse(data_set, walk_to(SplitDirection::LEFT, node_index), start,
                  mid);
    train_recurse(data_set, walk_to(SplitDirection::RIGHT, node_index), mid,
                  end);
  }

 private:
  std::vector<DecisionNode<Feature, Label, SplitterFn>> nodes_;
};
}
}
#endif /* TREE_H */
