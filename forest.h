#ifndef FOREST_H
#define FOREST_H

#include <vector>

#include "functional.h"
#include "threadpool.h"
#include "tree.h"

namespace qp {
namespace rf {

// A collection of decision trees which each cast a vote towards the final
// classificaiton of a sample.  Each tree is trained on a bagged (Bootstrap
// aggregated) sample of the original data set. Tree training is run in parallel
// using N threads, where N is the number of cores on the host machine.
//
// https://en.wikipedia.org/wiki/Bootstrap_aggregating
template <typename Feature, typename Label, typename SpiltterFn>
class DecisionForest {
 public:
  DecisionForest(std::size_t n_trees, std::size_t max_depth) {
    for (unsigned i = 0; i < n_trees; ++i) {
      trees_.emplace_back(max_depth);
    }
  }

  // Train each tree in parallel on a bagged sample of the dataset.
  void train(const DataSet<Feature, Label>& data_set) {
    std::vector<std::future<int>> futures;
    for (auto& tree : trees_) {
      futures.emplace_back(thread_pool_.add([&data_set, &tree]() {
        auto sample = sample_with_replacement(data_set, data_set.size());
        tree.train(sample);
        return 1;
      }));
    }

    // Thread pool futures are non-blocking.
    for (auto& fut : futures) {
      fut.wait();
    }
  }

  // Predict the label of a set of features.  This is done by predicting the
  // label using each of the trees in the forest, and then taking the majority
  // label over all trees.
  //
  // This could be parallelized, but running a feature vector through a tree
  // is generally very fast and thus unnecessary.
  Label predict(const std::vector<Feature>& features) {
    std::map<Label, int> predictions;
    for (const auto& tree : trees_) {
      ++predictions[tree.predict(features)];
    }

    auto prediction = std::max_element(predictions.begin(), predictions.end(),
                                       CompareOnSecond<Label, int>());
    return prediction->first;
  }

 private:
  std::vector<DecisionTree<Feature, Label, SpiltterFn>> trees_;
  threading::Threadpool<int> thread_pool_;
};

}  // namespace rf
}  // namespace qp

#endif /* FOREST_H */
