#ifndef SPLIT_FNS_H
#define SPLIT_FNS_H

#include "dataset.h"
#include "node.h"
#include "random.h"
#include "single_layer_perceptron.h"

namespace qp {
namespace rf {

// Splits on N features randomly.
// TODO Template for feauture and label.
// TODO Rethink this.  It doesn't work well with higher than 1 dimension.
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
class ModeVsAllPerceptronSplit {
 public:
  ModeVsAllPerceptronSplit() : layer_(N, 1, random_real_range<double>(0, 1)) {}

  void train(const qp::rf::SampledDataSet<Feature, Label>& data_set,
             std::size_t s, std::size_t e) {
    const auto total_features = data_set.front().get().features.size();
    generate_back_n(projection_, N, std::bind(random_range<FeatureIndex>, 0,
                                              total_features - 1));

    const auto should_fire =
        mode_label<Feature, Label>(data_set.begin() + s, data_set.begin() + e);
    const std::vector<double> fire = {1};
    const std::vector<double> not_fire = {-1};

    for (unsigned i = s; i < e; ++i) {
      const auto projected = project(data_set[i].get().features, projection_);
      layer_.learn(projected,
                   data_set[i].get().label == should_fire ? fire : not_fire);
    }
  }

  qp::rf::SplitDirection apply(const std::vector<Feature>& features) const {
    // This node was not reached during training.
    if (projection_.empty()) {
      generate_back_n(projection_, N, std::bind(random_range<FeatureIndex>, 0,
                                                features.size() - 1));
    }
    const auto projected = project(features, projection_);
    const auto output = layer_.predict(projected);
    return output.front() == 1 ? qp::rf::SplitDirection::LEFT
                               : qp::rf::SplitDirection::RIGHT;
  }

 private:
  SingleLayerPerceptron<Feature, Label, StepActivation> layer_;
  mutable std::vector<FeatureIndex> projection_;
};

double weight_update(const int& feature, const double error,
                     const double learning_rate, const double current_weight) {
  return current_weight + (learning_rate * error * feature);
}

// Use a single layer of perceptrons to train a split on N input features.
// The perceptron will find the most occuring label in the sample set, and
// attempt learn a split which segregates examples with that label.
template <typename Feature, typename Label, int N, int Iterations>
class PerceptronSplit {
 public:
  void train(const qp::rf::SampledDataSet<Feature, Label>& data_set,
             std::size_t s, std::size_t e) {
    // Generate random feature indicies, weights, bias, and learning rate.
    generate_back_n(feature_indicies_, N,
                    std::bind(random_range<FeatureIndex>, 0,
                              data_set.front().get().features.size() - 1));
    generate_back_n(weights_, N, std::bind(random_real_range<double>, -1, 1));
    bias_ = random_real_range<double>(-1, 1);
    learning_rate_ = random_real_range<double>(0, 1);

    // Find the most occuring label.
    const auto should_go_left =
        mode_label<Feature, Label>(data_set.begin() + s, data_set.begin() + e);

    // Show the perceptron training examples.  If the example is labeled with
    // the mode label, then learn a left split otherwise learn a right split.
    for (int it = 0; it < Iterations; ++it) {
      for (unsigned i = s; i < e; ++i) {
        show(data_set[i].get().features,
             data_set[i].get().label == should_go_left
                 ? qp::rf::SplitDirection::LEFT
                 : qp::rf::SplitDirection::RIGHT);
      }
    }
  }

  // Show the percentron the training example, and adjust the weights and bias
  // according to the error.
  void show(const std::vector<Feature>& features,
            qp::rf::SplitDirection label) {
    auto output = apply(features);
    adjust(features, static_cast<int>(label) - static_cast<int>(output));
  }

  // Adjust the weights and bias given a feature vector and an error.
  void adjust(const std::vector<Feature>& features, int error) {
    for (unsigned i = 0; i < weights_.size(); ++i) {
      weights_[i] = weight_update(features[feature_indicies_[i]], (double)error,
                                  learning_rate_, weights_[i]);
    }
    bias_ = weight_update(1.0, error, learning_rate_, bias_);
  }

  // Feed the features into the perceptrons and determine the split direction
  // based on if the output neuron fires.
  qp::rf::SplitDirection apply(const std::vector<Feature>& features) const {
    const auto projected = project(features, feature_indicies_);
    double sum = std::inner_product(weights_.begin(), weights_.end(),
                                    projected.begin(), -1 * bias_);
    return sum > 0 ? qp::rf::SplitDirection::LEFT
                   : qp::rf::SplitDirection::RIGHT;
  }

  const std::vector<FeatureIndex>& get_features() const {
    assert(!feature_indicies_.empty());
    return feature_indicies_;
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
