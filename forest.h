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
                 qp::threading::Threadpool* thread_pool,
                 TreeType tree_type = TreeType::SINGLE_FOREST)
      : thread_pool_(thread_pool) {
    trees_.reserve(n_trees);
    for (unsigned i = 0; i < n_trees; ++i) {
      trees_.emplace_back(max_depth, tree_type);
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
  std::vector<double> transform(const std::vector<double>& features) const {
    std::vector<double> transformed(trees_.size());
    for (auto i = 0UL; i < trees_.size(); ++i) {
      // transformed[i] =
      // trees_[i].walk(features)->mahalanobis_distance(features);
      transformed[i] = trees_[i].transform_summation(features);
    }
    return transformed;
  }

  // Transform an entire dataset of features.
  DataSet transform(const DataSet& data_set) const {
    auto transformed = empty_data_set(data_set.size(), trees_.size());
    for (auto sample = 0ul; sample < data_set.size(); ++sample) {
      // Carry over the same label.
      transformed[sample].label = data_set[sample].label;
      for (auto tree = 0ul; tree < trees_.size(); ++tree) {
        // Provide a transformed feature for each tree.
        /*const auto dist = trees_[tree]
                              .walk(data_set[sample].features)
                              ->mahalanobis_distance(data_set[sample].features);*/
        const auto dist =
            trees_[tree].transform_summation(data_set[sample].features);

        transformed[sample].features[tree] = dist;
      }
    }
    return transformed;
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

 private:
  std::vector<DecisionTree<SpiltterFn>> trees_;
  qp::threading::Threadpool* thread_pool_;
};

}  // namespace rf
}  // namespace qp

#endif /* FOREST_H */
