#ifndef DATASET_H
#define DATASET_H

#include <algorithm>
#include <set>
#include <unordered_map>
#include <vector>

#include "functional.h"
#include "random.h"
#include "vector_util.h"

namespace qp {
namespace rf {

// An example to be provided to a random forest.  Defined by a set of features
// and a label.
struct Example {
  std::vector<double> features;
  double label;
};

// An example that has been sampled from a dataset.  The reference wrapper
// provides fast copying when sampling large datasets.
using SampledExample = std::reference_wrapper<const Example>;

using FeatureIndex = std::size_t;

// A comparator which compares the i'th feature of two training examples using
// Cmp.
template <typename Cmp = std::less<double>>
class CompareOnFeature {
 public:
  CompareOnFeature(FeatureIndex i) : fx_(i) {}

  bool operator()(const SampledExample& lhs, const SampledExample& rhs) const {
    return cmp_(lhs.get().features[fx_], rhs.get().features[fx_]);
  }

  bool operator()(const Example& lhs, const Example& rhs) const {
    return cmp_(lhs.features[fx_], rhs.features[fx_]);
  }

 private:
  Cmp cmp_;
  FeatureIndex fx_;
};

// A dataset is a collection of training examples.
using DataSet = std::vector<Example>;

// A sampled dataset.  Again, fast copying.
using SampledDataSet = std::vector<SampledExample>;

using SDIter = SampledDataSet::iterator;

using LabelHistogram = std::unordered_map<double, std::size_t>;

// Generates an empty dataset with n_samples, each containing n_features.
DataSet empty_data_set(std::size_t n_samples, std::size_t n_features) {
  DataSet data_set(n_samples);
  for (auto& example : data_set) {
    example.features.resize(n_features);
  }
  return data_set;
}

// Sample n random items from the provided dataset with replacement.
SampledDataSet sample_with_replacement(const DataSet& data_set, std::size_t n) {
  SampledDataSet sample;
  const std::size_t total_examples = data_set.size();
  for (std::size_t i = 0; i < n; ++i) {
    sample.push_back(
        data_set[random_range<std::size_t>(0ul, total_examples - 1)]);
  }
  return sample;
}

// Creates a sampled dataset that contains exactly the elements of the source.
SampledDataSet sample_exactly(const DataSet& dataset) {
  SampledDataSet sample;
  for (const auto& example : dataset) {
    sample.push_back(example);
  }
  return sample;
}

// Find the most commonly occurring label in the dataset.
double mode_label(SDIter start, SDIter end) {
  LabelHistogram histogram;
  while (start != end) {
    ++histogram[start->get().label];
    ++start;
  }

  const auto mode = std::max_element(histogram.begin(), histogram.end(),
                                     CompareOnSecond<double, int>());
  return mode->first;
}

// Determines if the dataset contains a single label.
bool single_label(SDIter start, SDIter end) {
  const auto first_label = start->get().label;
  const auto equals_first_label = [&first_label](const auto& example) {
    return first_label == example.get().label;
  };

  return std::all_of(start, end, equals_first_label);
}

// Centers the dataset on a given mean vector.
void zero_center_mean(DataSet& dataset, const std::vector<double>& means) {
  for (auto& example : dataset) {
    vector_minus(example.features, means);
  }
}

// Centers the mean of the given dataset on 0.  Helps improve performance
// and convergence speed of perceptron splitters.  Returns the mean vector.
std::vector<double> zero_center_mean(DataSet& dataset) {
  const auto n_features = dataset.front().features.size();
  const auto n_samples_real = static_cast<double>(dataset.size());
  std::vector<double> means(n_features, 0);

  for (const auto& example : dataset) {
    for (auto feature = 0ul; feature < n_features; ++feature) {
      means[feature] += example.features[feature] / n_samples_real;
    }
  }
  zero_center_mean(dataset, means);
  return means;
}

}  // namespace rf
}  // namespace qp
#endif /* DATASET_H */
