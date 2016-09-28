#ifndef SPLIT_FNS_H
#define SPLIT_FNS_H

#include "dataset.h"
#include "node.h"
#include "random.h"

namespace qp {
namespace rf {

// Splits on N features randomly.
// TODO Template for feauture and label.
template <int N>
class NDimensionalSplit {
 public:
  void train(const qp::rf::SampledDataSet<int, int>& set, std::size_t s,
             std::size_t e) {
    for (int i = 0; i < N; ++i) {
      auto feature_index =
          qp::rf::random_range<int>(0ul, set.front().get().features.size());
      auto feature_range = std::minmax_element(
          set.begin() + s, set.begin() + e,
          qp::rf::CompareOnFeature<int, int>(feature_index));
      auto low = feature_range.first->get().features[feature_index];
      auto high = feature_range.second->get().features[feature_index];
      auto threshold = qp::rf::random_range<int>(low, high);

      feature_indexes.emplace_back(feature_index);
      thresholds.emplace_back(threshold);
    }
  }

  qp::rf::SplitDirection apply(const std::vector<int>& features) const {
    for (unsigned i = 0; i < feature_indexes.size(); ++i) {
      if (features[feature_indexes[i]] < thresholds[i]) {
        return qp::rf::SplitDirection::LEFT;
      }
    }
    return qp::rf::SplitDirection::RIGHT;
  }

 private:
  std::vector<int> feature_indexes;
  std::vector<int> thresholds;
};

template <int N>
class PerceptronSplit {
 public:
  void train(const qp::rf::SampledDataSet<int, int>& data_set, std::size_t s,
             std::size_t e) {
    for (int i = 0; i < N; ++i) {
      feature_indicies.push_back(
          random_range<int>(0, data_set.front().get().features.size()));
      weights.push_back(random_real_range<double>(-1, 1));
    }
    threshold = random_real_range<double>(-1, 1);
    learning_rate = random_real_range<double>(0, 1);

    auto should_go_left =
        mode_on_label<int, int>(data_set.begin() + s, data_set.begin() + e);
    for (unsigned i = s; i < e; ++i) {
      show(data_set[i].get().features, data_set[i].get().label == should_go_left
                                           ? qp::rf::SplitDirection::LEFT
                                           : qp::rf::SplitDirection::RIGHT);
    }
  }

  void show(const std::vector<int>& features, qp::rf::SplitDirection label) {
    auto output = apply(features);
    adjust(features, static_cast<int>(label) - static_cast<int>(output));
  }

  void adjust(const std::vector<int>& features, int error) {
    for (unsigned i = 0; i < weights.size(); ++i) {
      weights[i] =
          weights[i] + (learning_rate * error * features[feature_indicies[i]]);
    }
    threshold = threshold + (learning_rate * error);
  }

  qp::rf::SplitDirection apply(const std::vector<int>& features) const {
    double sum = 0;
    for (unsigned i = 0; i < feature_indicies.size(); ++i) {
      sum += features[feature_indicies[i]] * weights[i];
    }

    return sum > threshold ? qp::rf::SplitDirection::LEFT
                           : qp::rf::SplitDirection::RIGHT;
  }

 private:
  std::vector<int> feature_indicies;
  std::vector<double> weights;
  double threshold;
  double learning_rate;
};

}  // namespace rf
}  // namespace qp

#endif /* SPLIT_FNS_H */
