#ifndef SPLIT_FNS_H
#define SPLIT_FNS_H

#include "dataset.h"
#include "node.h"
#include "random.h"
#include "single_layer_perceptron.h"

namespace qp {
namespace rf {

// This class is totally symbolic.  Split functions should conform to this
// interface.
template <typename Feature, typename Label>
class SplitFunction {
  virtual void train(const qp::rf::SampledDataSet<Feature, Label>&, std::size_t,
                     std::size_t) = 0;
  virtual qp::rf::SplitDirection apply(const std::vector<Feature>&) const = 0;
  virtual const std::vector<FeatureIndex>& get_features() const = 0;
  virtual std::size_t n_input_features() const = 0;
};

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
    const auto threshold = qp::rf::random_real_range<Feature>(low, high);
  }

  qp::rf::SplitDirection apply(const std::vector<Feature>& features) const {
    return features[feature_index_] < threshold_
               ? qp::rf::SplitDirection::LEFT
               : qp::rf::SplitDirection::RIGHT;
  }

  const std::vector<FeatureIndex> get_features() const {
    return {feature_index_};
  }

  std::size_t n_input_features() const { return 1; }

 private:
  FeatureIndex feature_index_;
  Feature threshold_;
};

template <typename Feature, typename Label, int N>
class RandomMultivariateSplit {
 public:
  void train(const qp::rf::SampledDataSet<Feature, Label>& set, std::size_t s,
             std::size_t e) {
    const auto total_features = set.front().get().features.size();
    // feature_index_ = random_range<FeatureIndex>(0, total_features - 1);
    generate_back_n(feature_indicies_, N, [&]() {
      return random_range<FeatureIndex>(0, total_features - 1);
    });

    thresholds_.resize(N);
    std::transform(feature_indicies_.begin(), feature_indicies_.end(),
                   thresholds_.begin(), [&](const FeatureIndex i) {
                     const auto feature_range = std::minmax_element(
                         set.begin() + s, set.begin() + e,
                         qp::rf::CompareOnFeature<Feature, Label>(i));
                     const auto low = feature_range.first->get().features[i];
                     const auto high = feature_range.second->get().features[i];
                     const auto threshold =
                         qp::rf::random_real_range<Feature>(low, high);
                     return threshold;

                   });
  }

  qp::rf::SplitDirection apply(const std::vector<Feature>& features) const {
    for (unsigned i = 0; i < feature_indicies_.size(); ++i) {
      if (features[feature_indicies_[i]] < thresholds_[i]) {
        return qp::rf::SplitDirection::LEFT;
      }
    }
    return qp::rf::SplitDirection::RIGHT;
  }

  const std::vector<FeatureIndex>& get_features() const {
    return feature_indicies_;
  }

  std::size_t n_input_features() const { return N; }

 private:
  std::vector<FeatureIndex> feature_indicies_;
  std::vector<Feature> thresholds_;
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

    std::vector<Feature> projected(N);
    for (unsigned i = s; i < e; ++i) {
      project(data_set[i].get().features, projection_, projected.begin());
      layer_.learn(projected,
                   data_set[i].get().label == should_fire ? fire : not_fire);
    }
  }

  qp::rf::SplitDirection apply(const std::vector<Feature>& features) const {
    const auto projected = project(features, projection_);
    const auto output = layer_.predict(projected);
    return output.front() == 1 ? qp::rf::SplitDirection::LEFT
                               : qp::rf::SplitDirection::RIGHT;
  }

  const std::vector<FeatureIndex>& get_features() const { return projection_; }

  std::size_t n_input_features() const { return N; }

 private:
  SingleLayerPerceptron<Feature, Label, StepActivation> layer_;
  std::vector<FeatureIndex> projection_;
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
    generate_back_n(projection_, N, [total_features]() {
      return random_range<FeatureIndex>(0, total_features - 1);
    });

    auto label_ids = label_identifiers(data_set, s, e);
    layer_.reset(new SingleLayerPerceptron<Feature, Label, SigmoidActivation>(
        N, label_ids.size(), random_real_range<double>(0, 1)));

    // First pass train the perceptron.
    std::vector<double> expected_output(label_ids.size(), 0);
    std::vector<Feature> projected(N);
    for (auto i = s; i != e; ++i) {
      expected_output[label_ids[data_set[i].get().label]] = 1;
      project(data_set[i].get().features, projection_, projected.begin());
      layer_->learn(projected, expected_output);
      expected_output[label_ids[data_set[i].get().label]] = 0;
    }

    // Second pass determine which output neuron contains the maximum average
    // activation value.
    std::vector<double> average_activations(label_ids.size(), 0);
    double n_samples_real = static_cast<double>(e - s + 1);
    for (auto i = s; i != e; ++i) {
      project(data_set[i].get().features, projection_, projected.begin());
      const auto output = layer_->predict(projected);
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
    const auto projected = project(features, projection_);
    const auto output = layer_->predict(projected);
    return output[maximum_activation_neuron] >= 0.5
               ? qp::rf::SplitDirection::LEFT
               : qp::rf::SplitDirection::RIGHT;
  }

  const std::vector<FeatureIndex>& get_features() const { return projection_; }

  std::size_t n_input_features() const { return N; }

 private:
  std::unique_ptr<SingleLayerPerceptron<Feature, Label, SigmoidActivation>>
      layer_;
  std::size_t maximum_activation_neuron;
  mutable std::vector<FeatureIndex> projection_;
};

}  // namespace rf
}  // namespace qp

#endif /* SPLIT_FNS_H */
