#ifndef SPLIT_FNS_H
#define SPLIT_FNS_H

#include <experimental/optional>
#include <map>

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

    // Randomly select N features.
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

  std::size_t n_input_features() const { return N; }

 private:
  std::vector<FeatureIndex> feature_indices_;
  SingleLayerPerceptron<Step> line_;
};

// Trains a perceptron in a mode-vs-all fashion, and splits based on the
// predicted outcome.  This split function should be paired with the Step
// activation function, but it is left as a parameter for experimentation.
template <typename Activation, int N>
class ModeVsAllPerceptronSplit {
 public:
  ModeVsAllPerceptronSplit() : layer_(N, 1, random_real_range<double>(0, 1)) {}

  void train(SDIter first, SDIter last) {
    // Randomly select features.
    const auto total_features = first->get().features.size();
    generate_back_n(projection_, N, std::bind(random_range<FeatureIndex>, 0,
                                              total_features - 1));

    // Determine the mode laabel.
    const auto should_fire = mode_label(first, last);
    const std::vector<double> fire = {layer_.maximum_activation()};
    const std::vector<double> not_fire = {layer_.minimum_activation()};

    std::vector<double> input_buffer(N);
    for (auto example = first; example != last; ++example) {
      project(example->get().features, projection_, input_buffer.begin());
      // If the example has the mode label then the perceptron should fire,
      // otherwise it should not.
      layer_.learn(input_buffer,
                   example->get().label == should_fire ? fire : not_fire);
    }
  }

  qp::rf::SplitDirection apply(const std::vector<double>& features) const {
    const auto input_buffer = project(features, projection_);
    const auto output_buffer = layer_.predict(input_buffer);
    return output_buffer.front() > layer_.fire_threshold()
               ? qp::rf::SplitDirection::LEFT
               : qp::rf::SplitDirection::RIGHT;
  }

  std::size_t n_input_features() const { return N; }

  double activate(const std::vector<double>& features) const {
    return layer_.predict(features).front();
  }

 private:
  SingleLayerPerceptron<Activation> layer_;
  std::vector<FeatureIndex> projection_;
};

// Selects a random contiguous block of features instead of randomly distributed
// ones.  The idea is that this will be more meaniningful for sequenced data.
template <typename Activation, int BlockSize>
class ModeVsAllBlockPerceptronSplit {
 public:
  ModeVsAllBlockPerceptronSplit()
      : layer_(BlockSize, 1, random_real_range<double>(0, 1)),
        block_buffer_(BlockSize) {}

  void load_block(const std::vector<double>& features,
                  std::vector<double>& buffer) const {
    std::copy(features.begin() + block_start_,
              features.begin() + block_start_ + BlockSize, buffer.begin());
  }

  void train(SDIter first, SDIter last) {
    const auto total_features = first->get().features.size();
    block_start_ =
        random_range<FeatureIndex>(0, total_features - 1 - BlockSize);

    const auto should_fire = mode_label(first, last);
    const std::vector<double> fire = {layer_.maximum_activation()};
    const std::vector<double> not_fire = {layer_.minimum_activation()};

    for (auto example = first; example != last; ++example) {
      load_block(example->get().features, block_buffer_);
      layer_.learn(block_buffer_,
                   example->get().label == should_fire ? fire : not_fire);
    }
  }

  qp::rf::SplitDirection apply(const std::vector<double>& features) const {
    load_block(features, block_buffer_);
    const auto output = layer_.predict(block_buffer_);
    return output.front() > layer_.fire_threshold()
               ? qp::rf::SplitDirection::LEFT
               : qp::rf::SplitDirection::RIGHT;
  }

  // TODO: this needs to be scaled.  Since the number of blocks of size N is
  // much smaller than the number of feature combinations of size N, using
  // BlockSize results in a lot of redundancy and increased training times.
  std::size_t n_input_features() const { return BlockSize; }

 private:
  mutable std::vector<double> block_buffer_;
  SingleLayerPerceptron<Activation> layer_;
  std::size_t block_start_;
};

// Trains a perceptron in a one-vs-one manner, and then determines which class
// produces the highest average activation value.  The activation of
// that class is then used as the split criteria.
template <typename Activation, int N>
class HighestAverageActivation {
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
    // Randomly select features.
    const auto total_features = first->get().features.size();
    generate_back_n(projection_, N, [total_features]() {
      return random_range<FeatureIndex>(0, total_features - 1);
    });

    auto label_ids = label_identifiers(first, last);
    layer_.reset(new SingleLayerPerceptron<Activation>(
        N, label_ids.size(), random_real_range<double>(0, 1)));

    // First pass train the perceptron.
    std::vector<double> expected_output(label_ids.size(),
                                        layer_->minimum_activation());
    std::vector<double> projected(N);
    for (auto example = first; example != last; ++example) {
      const auto label_id = label_ids[example->get().label];
      expected_output[label_id] = layer_->maximum_activation();
      project(example->get().features, projection_, projected.begin());
      layer_->learn(projected, expected_output);
      expected_output[label_id] = layer_->minimum_activation();
    }

    // Second pass determine which output neuron contains the maximum average
    // activation value.
    std::vector<double> average_activations(label_ids.size(), 0);
    double n_samples_real = static_cast<double>(last - first + 1);
    for (auto example = first; example != last; ++example) {
      project(example->get().features, projection_, projected.begin());
      const auto output = layer_->predict(projected);
      for (auto activation = 0ul; activation < output.size(); ++activation) {
        average_activations[activation] += output[activation];
      }
    }

    for (auto& activation : average_activations) {
      activation /= n_samples_real;
    }

    const auto max = std::max_element(average_activations.begin(),
                                      average_activations.end());
    maximum_activation_neuron_ = max - average_activations.begin();
  }

  // Determine the split direction based on the output of the maximum activation
  // neuron determined during the activation pass.
  qp::rf::SplitDirection apply(const std::vector<double>& features) const {
    const auto projected = project(features, projection_);
    const auto output = layer_->predict(projected);
    return output[maximum_activation_neuron_] > layer_->fire_threshold()
               ? qp::rf::SplitDirection::LEFT
               : qp::rf::SplitDirection::RIGHT;
  }

  double activate(const std::vector<double>& features) const {
    return layer_->predict(
        project(features, projection_))[maximum_activation_neuron_];
  }

  std::size_t n_input_features() const { return N; }

 private:
  std::unique_ptr<SingleLayerPerceptron<Activation>> layer_;
  std::size_t maximum_activation_neuron_;
  std::vector<FeatureIndex> projection_;
};

// Chooses a random split function from all of the above.
// Note that the activation parameter is only actually used if a perceptron
// based split function is selected, and the N parameter is ignored if the
// random univariate splitter is selected.
template <typename Activation, int N>
class RandomSplitFunction {
 public:
  template <typename T>
  using Maybe = std::experimental::optional<T>;

  void train(SDIter first, SDIter last) {
    const int random = random_range(0, 3);
    if (random == 0) {
      split_fn_1 = RandomUnivariateSplit();
    } else if (random == 1) {
      split_fn_2 = RandomMultivariateSplit<N>();
    } else if (random == 2) {
      split_fn_3 = ModeVsAllPerceptronSplit<Activation, N>();
    } else {
      split_fn_4 = HighestAverageActivation<Activation, N>();
    }

    if (split_fn_1) split_fn_1->train(first, last);
    if (split_fn_2) split_fn_2->train(first, last);
    if (split_fn_3) split_fn_3->train(first, last);
    if (split_fn_4) split_fn_4->train(first, last);
  }

  qp::rf::SplitDirection apply(const std::vector<double>& features) const {
    if (split_fn_1) return split_fn_1->apply(features);
    if (split_fn_2) return split_fn_2->apply(features);
    if (split_fn_3) return split_fn_3->apply(features);
    if (split_fn_4) return split_fn_4->apply(features);
    assert(false);
  }

  std::size_t n_input_features() const { return N; }

 private:
  // TODO: once std::variant becomes standardized used that.
  Maybe<RandomUnivariateSplit> split_fn_1;
  Maybe<RandomMultivariateSplit<N>> split_fn_2;
  Maybe<ModeVsAllPerceptronSplit<Activation, N>> split_fn_3;
  Maybe<HighestAverageActivation<Activation, N>> split_fn_4;
};

}  // namespace rf
}  // namespace qp

#endif /* SPLIT_FNS_H */
