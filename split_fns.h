#ifndef SPLIT_FNS_H
#define SPLIT_FNS_H

#include "dataset.h"
#include "node.h"
#include "random.h"
#include "single_layer_perceptron.h"

namespace qp {
namespace rf {

template <typename Feature, typename Label>
class RandomUnivariateSplit {
 public:
  void train(const qp::rf::SampledDataSet<Feature, Label>& set, std::size_t s,
             std::size_t e) {
    const auto total_features = set.front().get().features.size();
    feature_index_ = random_range<FeatureIndex>(0, total_features - 1);

    const auto feature_range = std::minmax_element(
        set.begin() + s, set.begin() + e,
        qp::rf::CompareOnFeature<Feature, Label>(feature_index_));

    const auto low = feature_range.first->get().features[feature_index_];
    const auto high = feature_range.second->get().features[feature_index_];
    const auto threshold = qp::rf::random_range<Feature>(low, high);
  }

  qp::rf::SplitDirection apply(const std::vector<Feature>& features) const {
    return features[feature_index_] < threshold_
               ? qp::rf::SplitDirection::LEFT
               : qp::rf::SplitDirection::RIGHT;
  }

  const std::vector<FeatureIndex>& get_features() const {
    return {feature_index_};
  }

 private:
  FeatureIndex feature_index_;
  Feature threshold_;
};

// Splits on N features randomly.
// TODO Template for feauture and label.
// TODO Rethink this.  It doesn't work well with higher than 1 dimension.
template <int N>
class NDimensionalSplit {
 public:
  void train(const qp::rf::SampledDataSet<int, int>& set, std::size_t s,
             std::size_t e) {
    for (int i = 0; i < N; ++i) {
      auto feature_index = qp::rf::random_range<FeatureIndex>(
          0ul, set.front().get().features.size());
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

  const std::vector<FeatureIndex>& get_features() const {
    return feature_indexes;
  }

 private:
  std::vector<FeatureIndex> feature_indexes;
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

  const std::vector<FeatureIndex>& get_features() const { return projection_; }

 private:
  SingleLayerPerceptron<Feature, Label, StepActivation> layer_;
  mutable std::vector<FeatureIndex> projection_;
};

template <typename Feature, typename Label, int N>
class HighestAverageSigmoidActivation {
 public:
  // Assign each label an incremental integer identifier.
  std::map<Label, int> label_identifiers(
      const SampledDataSet<Feature, Label>& data_set, std::size_t s,
      std::size_t e) const {
    std::map<Label, int> ids;
    int current_id = 0;
    for (auto i = s; i != e; ++i) {
      const auto check = ids.insert({data_set[i].get().label, current_id});
      if (check.second) ++current_id;
    }
    return ids;
  }

  void train(const qp::rf::SampledDataSet<Feature, Label>& data_set,
             std::size_t s, std::size_t e) {
    const auto total_features = data_set.front().get().features.size();
    generate_back_n(projection_, N, std::bind(random_range<FeatureIndex>, 0,
                                              total_features - 1));

    auto label_ids = label_identifiers(data_set, s, e);
    layer_.reset(new SingleLayerPerceptron<Feature, Label, SigmoidActivation>(
        N, label_ids.size(), random_real_range<double>(0, 1)));

    // First pass train the perceptron.
    std::vector<double> expected_output(label_ids.size(), 0);
    for (auto i = s; i != e; ++i) {
      expected_output[label_ids[data_set[i].get().label]] = 1;
      const auto projected = project(data_set[i].get().features, projection_);
      layer_->learn(projected, expected_output);
      expected_output[label_ids[data_set[i].get().label]] = 0;
    }

    // Second pass determine which output neuron contains the maximum average
    // activation value.
    std::vector<double> average_activations(label_ids.size(), 0);
    std::vector<double> output;
    double n_samples_real = static_cast<double>(e - s + 1);
    for (auto i = s; i != e; ++i) {
      const auto projected = project(data_set[i].get().features, projection_);
      output = layer_->predict(projected);
      for (auto activation = 0ul; activation < output.size(); ++activation) {
        average_activations[activation] += output[activation] / n_samples_real;
      }
    }

    const auto max = std::max_element(average_activations.begin(),
                                      average_activations.end());
    for (auto i = 0ul; i < average_activations.size(); ++i) {
      if (average_activations[i] == *max) {
        maximum_activation_neuron = i;
        break;
      }
    }
  }

  qp::rf::SplitDirection apply(const std::vector<Feature>& features) const {
    // This node was not reached during training.
    if (projection_.empty()) {
      return qp::rf::SplitDirection::LEFT;
      // generate_back_n(projection_, N, std::bind(random_range<FeatureIndex>,
      // 0,
      // features.size() - 1));
    }
    const auto projected = project(features, projection_);
    const auto output = layer_->predict(projected);
    return output[maximum_activation_neuron] >= 0.5
               ? qp::rf::SplitDirection::LEFT
               : qp::rf::SplitDirection::RIGHT;
  }

  const std::vector<FeatureIndex>& get_features() const { return projection_; }

 private:
  std::unique_ptr<SingleLayerPerceptron<Feature, Label, SigmoidActivation>>
      layer_;
  std::size_t maximum_activation_neuron;
  mutable std::vector<FeatureIndex> projection_;
};

}  // namespace rf
}  // namespace qp

#endif /* SPLIT_FNS_H */
