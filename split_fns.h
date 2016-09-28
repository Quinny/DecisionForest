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

template <typename Feature, typename Label, int N>
class PerceptronSplit {
 public:
  void train(const qp::rf::SampledDataSet<Feature, Label>& data_set,
             std::size_t s, std::size_t e) {
    for (int i = 0; i < N; ++i) {
      feature_indicies_.push_back(random_range<FeatureIndex>(
          0, data_set.front().get().features.size()));
      weights_.push_back(random_real_range<double>(-1, 1));
    }
    bias_ = random_real_range<double>(-1, 1);
    learning_rate_ = random_real_range<double>(0, 1);

    const auto should_go_left =
        mode_label<Feature, Label>(data_set.begin() + s, data_set.begin() + e);
    for (unsigned i = s; i < e; ++i) {
      show(data_set[i].get().features, data_set[i].get().label == should_go_left
                                           ? qp::rf::SplitDirection::LEFT
                                           : qp::rf::SplitDirection::RIGHT);
    }
  }

  void show(const std::vector<Feature>& features,
            qp::rf::SplitDirection label) {
    auto output = apply(features);
    adjust(features, static_cast<int>(label) - static_cast<int>(output));
  }

  void adjust(const std::vector<Feature>& features, int error) {
    for (unsigned i = 0; i < weights_.size(); ++i) {
      weights_[i] = weights_[i] +
                    (learning_rate_ * error * features[feature_indicies_[i]]);
    }
    bias_ = bias_ + (learning_rate_ * error);
  }

  qp::rf::SplitDirection apply(const std::vector<Feature>& features) const {
    double sum = 0;
    for (unsigned i = 0; i < feature_indicies_.size(); ++i) {
      sum += features[feature_indicies_[i]] * weights_[i];
    }

    return sum > bias_ ? qp::rf::SplitDirection::LEFT
                       : qp::rf::SplitDirection::RIGHT;
  }

 private:
  std::vector<FeatureIndex> feature_indicies_;
  std::vector<double> weights_;
  double bias_;
  double learning_rate_;
};

}  // namespace rf
}  // namespace qp

#endif /* SPLIT_FNS_H */
