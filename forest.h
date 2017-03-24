#ifndef FOREST_H
#define FOREST_H

#include <vector>

#include "functional.h"
#include "logging.h"
#include "threadpool.h"
#include "tree.h"

namespace qp {
namespace rf {

// A collection of decision trees which each cast a vote towards the final
// classification of a sample. Tree training is done on the provided thread
// pool.
template <typename SpiltterFn>
class DecisionForest {
 public:
  // Grow a forest of size |n_trees|, each of depth |max_depth|. Passing -1 as a
  // the max_depth will cause the tree to be fully grown.  TreeType defines
  // whether the forest will be used in a deep forest or not (deep forest's
  // perform extra computations not needed within single forests).
  DecisionForest(std::size_t n_trees, std::size_t max_depth,
                 qp::threading::Threadpool* thread_pool, int leaf_threshold = 1,
                 TreeType tree_type = TreeType::SINGLE_FOREST)
      : thread_pool_(thread_pool) {
    trees_.reserve(n_trees);
    for (unsigned i = 0; i < n_trees; ++i) {
      trees_.emplace_back(max_depth, leaf_threshold, tree_type);
    }
  }

  // Trains each tree in the forest on the provided dataset.  Tree training is
  // done in parallel on the provided thread pool.
  void train(const DataSet& data_set) {
    qp::ProgressBar progress(trees_.size());

    std::vector<std::future<void>> futures;
    futures.reserve(trees_.size());
    for (auto& tree : trees_) {
      futures.emplace_back(thread_pool_->add([&data_set, &tree, this]() {
        auto sample = sample_exactly(data_set);
        tree.train(sample);
      }));
    }

    // Thread pool futures are non-blocking.
    for (auto& fut : futures) {
      fut.wait();
      progress.progress(1);
    }
  }

  // Transform the vector of features by computing the mahalanobis distance
  // to the leaf node in each tree which would classify the data.
  // The resulting vector will be 1 x |trees|.
  void transform(std::vector<double>& features) const {
    // TODO: does this matter? I think not since this tree was only trained on
    // non-augumented features therefore it should never consider the newly
    // added ones.
    for (auto i = 0UL; i < trees_.size(); ++i) {
      features.push_back(trees_[i].transform_summation(features));
    }
  }

  // Transform an entire dataset of features.
  void transform(DataSet& data_set) const {
    for (auto sample = 0ul; sample < data_set.size(); ++sample) {
      transform(data_set[sample].features);
    }
  }

  // Predict the label of a set of features.  This is done by predicting the
  // label using each of the trees in the forest, and then taking the majority
  // label over all trees.
  double predict(const std::vector<double>& features) {
    LabelHistogram predictions;
    for (const auto& tree : trees_) {
      ++predictions[tree.predict(features)];
    }

    auto prediction = std::max_element(predictions.begin(), predictions.end(),
                                       CompareOnSecond<double, std::size_t>());
    return prediction->first;
  }

  double average_depth() const {
    int sum = 0;
    for (const auto& tree : trees_) {
      sum += tree.depth();
    }
    return sum / static_cast<double>(trees_.size());
  }

 private:
  std::vector<DecisionTree<SpiltterFn>> trees_;
  qp::threading::Threadpool* thread_pool_;
};

}  // namespace rf
}  // namespace qp

#endif /* FOREST_H */
