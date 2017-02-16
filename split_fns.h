#ifndef SPLIT_FNS_H
#define SPLIT_FNS_H

#include "dataset.h"
#include "node.h"
#include "random.h"
#include "single_layer_perceptron.h"

/*
 * A collection of split functions to use as weak learners inside of decision
 * trees.
 */

namespace qp {
namespace rf {

// This class is totally symbolic.  Split functions should conform to this
// interface.
class SplitFunction {
  virtual void train(SDIter, SDIter) = 0;
  virtual qp::rf::SplitDirection apply(const std::vector<double>&) const = 0;
  virtual const std::vector<FeatureIndex>& get_features() const = 0;
  virtual std::size_t n_input_features() const = 0;
};

// Typical random univariate split, choose a feature and a random threshold
// and split on that.
class RandomUnivariateSplit {
 public:
  void train(SDIter first, SDIter last) {
    const auto total_features = first->get().features.size();
    feature_index_ = random_range<FeatureIndex>(0, total_features - 1);

    const auto feature_range = std::minmax_element(
        first, last, qp::rf::CompareOnFeature<>(feature_index_));

    const auto low = feature_range.first->get().features[feature_index_];
    const auto high = feature_range.second->get().features[feature_index_];
    threshold_ = qp::rf::random_real_range<double>(low, high);
  }

  qp::rf::SplitDirection apply(const std::vector<double>& features) const {
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
  double threshold_;
};

// Splits based on the sign of the dot product of a projection of the provided
// feature vector and a random N dimensional line.
template <int N>
class RandomMultivariateSplit {
 public:
  // An untrained single layer step activated perceptron is used as the
  // random line.
  RandomMultivariateSplit() : line_(N, 1, 0) {}

  void train(SDIter first, SDIter last) {
    (void)last;

    const auto total_features = first->get().features.size();
    generate_back_n(feature_indices_, N, [&]() {
      return random_range<FeatureIndex>(0, total_features - 1);
    });
  }

  qp::rf::SplitDirection apply(const std::vector<double>& features) const {
    return line_.predict(project(features, feature_indices_)).front() == 1
               ? qp::rf::SplitDirection::LEFT
               : qp::rf::SplitDirection::RIGHT;
  }

  const std::vector<FeatureIndex>& get_features() const {
    return feature_indices_;
  }

  std::size_t n_input_features() const { return N; }

 private:
  std::vector<FeatureIndex> feature_indices_;
  SingleLayerPerceptron<StepActivation> line_;
};

// Trains a perceptron in a mode-vs-all fashion, and splits based on the
// predicted outcome.
template <int N>
class ModeVsAllPerceptronSplit {
 public:
  ModeVsAllPerceptronSplit() : layer_(N, 1, random_real_range<double>(0, 1)) {}

  void train(SDIter first, SDIter last) {
    const auto total_features = first->get().features.size();
    generate_back_n(projection_, N, std::bind(random_range<FeatureIndex>, 0,
                                              total_features - 1));

    const auto should_fire = mode_label(first, last);
    const std::vector<double> fire = {1};
    const std::vector<double> not_fire = {-1};

    std::vector<double> projected(N);
    for (auto i = first; i != last; ++i) {
      project(i->get().features, projection_, projected.begin());
      layer_.learn(projected, i->get().label == should_fire ? fire : not_fire);
    }
  }

  qp::rf::SplitDirection apply(const std::vector<double>& features) const {
    const auto projected = project(features, projection_);
    const auto output = layer_.predict(projected);
    return output.front() == 1 ? qp::rf::SplitDirection::LEFT
                               : qp::rf::SplitDirection::RIGHT;
  }

  const std::vector<FeatureIndex>& get_features() const { return projection_; }

  std::size_t n_input_features() const { return N; }

 private:
  SingleLayerPerceptron<StepActivation> layer_;
  std::vector<FeatureIndex> projection_;
};

// Trains a perceptron in a one-vs-one manner, and then determines which class
// produces the highest average sigmoid activation value.  The activation of
// that class is then used as the split criteria.
template <int N>
class HighestAverageSigmoidActivation {
 public:
  // Assign each label an incremental integer identifier.
  std::map<double, int> label_identifiers(SDIter first, SDIter last) const {
    std::map<double, int> ids;
    int current_id = 0;
    for (auto i = first; i != last; ++i) {
      const auto check = ids.insert({i->get().label, current_id});
      if (check.second) ++current_id;
    }
    return ids;
  }

  void train(SDIter first, SDIter last) {
    const auto total_features = first->get().features.size();
    generate_back_n(projection_, N, [total_features]() {
      return random_range<FeatureIndex>(0, total_features - 1);
    });

    auto label_ids = label_identifiers(first, last);
    layer_.reset(new SingleLayerPerceptron<SigmoidActivation>(
        N, label_ids.size(), random_real_range<double>(0, 1)));

    // First pass train the perceptron.
    std::vector<double> expected_output(label_ids.size(), 0);
    std::vector<double> projected(N);
    for (auto i = first; i != last; ++i) {
      const auto label_id = label_ids[i->get().label];
      expected_output[label_id] = 1;
      project(i->get().features, projection_, projected.begin());
      layer_->learn(projected, expected_output);
      expected_output[label_id] = 0;
    }

    // Second pass determine which output neuron contains the maximum average
    // activation value.
    std::vector<double> average_activations(label_ids.size(), 0);
    double n_samples_real = static_cast<double>(last - first + 1);
    for (auto i = first; i != last; ++i) {
      project(first->get().features, projection_, projected.begin());
      const auto output = layer_->predict(projected);
      for (auto activation = 0ul; activation < output.size(); ++activation) {
        average_activations[activation] += output[activation] / n_samples_real;
      }
    }

    const auto max = std::max_element(average_activations.begin(),
                                      average_activations.end());
    maximum_activation_neuron_ = max - average_activations.begin();
  }

  qp::rf::SplitDirection apply(const std::vector<double>& features) const {
    const auto projected = project(features, projection_);
    const auto output = layer_->predict(projected);
    return output[maximum_activation_neuron_] >= 0.5
               ? qp::rf::SplitDirection::LEFT
               : qp::rf::SplitDirection::RIGHT;
  }

  double activate(const std::vector<double>& features) const {
    return layer_->predict(
        project(features, projection_))[maximum_activation_neuron_];
  }

  const std::vector<FeatureIndex>& get_features() const { return projection_; }

  std::size_t n_input_features() const { return N; }

 private:
  std::unique_ptr<SingleLayerPerceptron<SigmoidActivation>> layer_;
  std::size_t maximum_activation_neuron_;
  std::vector<FeatureIndex> projection_;
};

}  // namespace rf
}  // namespace qp

#endif /* SPLIT_FNS_H */
