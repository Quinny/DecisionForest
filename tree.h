#ifndef TREE_H
#define TREE_H

#include <algorithm>
#include <cmath>
#include "dataset.h"
#include "node.h"

namespace qp {
namespace rf {

template <typename Feature, typename Label, typename SplitterFn>
class DecisionTree {
 public:
  DecisionTree(std::size_t depth) : nodes_(std::pow(2, depth + 1) - 1) {}

  bool is_leaf(std::size_t i) const { return 2 * i + 1 >= nodes_.size(); }

  std::size_t walk_to(SplitDirection dir, std::size_t i) const {
    return dir == SplitDirection::LEFT ? 2 * i + 1 : 2 * i + 2;
  }

  Label predict(const std::vector<Feature>& features) const {
    std::size_t node_index = 0;

    while (!is_leaf(node_index)) {
      node_index =
          walk_to(nodes_[node_index].split_direction(features), node_index);
    }
    return nodes_[node_index].predict();
  }

  void train(SampledDataSet<Feature, Label> data_set) {
    train_recurse(data_set, /* node_index=*/0, /* start=*/0,
                  /* end=*/data_set.size());
  }

  void train_recurse(SampledDataSet<Feature, Label>& data_set,
                     std::size_t node_index, std::size_t start,
                     std::size_t end) {
    if (node_index >= nodes_.size() || start >= end) {
      return;
    }

    nodes_[node_index].train(data_set, start, end);

    auto pivot_iter = std::partition(
        data_set.begin() + start, data_set.begin() + end,
        [&](const auto& sample) {
          return nodes_[node_index].split_direction(sample.get().features) ==
                 SplitDirection::LEFT;
        });

    auto mid = std::distance(data_set.begin(), pivot_iter);

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
