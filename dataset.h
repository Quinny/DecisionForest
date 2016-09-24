#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include "random.h"

namespace qp {
namespace rf {

// An example to be provided to a random forest.  Defined by a set of features
// and a label.  For now, only integral label and feature types are allowed.
template <typename Feature, typename Label>
struct Example {
  std::vector<Feature> features;
  Label label;
};

// An example that has been sampled from a dataset.
template <typename Feature, typename Label>
using SampledExample = std::reference_wrapper<const Example<Feature, Label>>;

// A comparator which compares the i'th feature of two sampled examples using
// Cmp.
template <typename Feature, typename Label, typename Cmp = std::less<Feature>>
class CompareOnFeature {
 public:
  CompareOnFeature(std::size_t i) : fx(i) {}

  bool operator()(const SampledExample<Feature, Label>& lhs,
                  const SampledExample<Feature, Label>& rhs) const {
    return cmp(lhs.get().features[fx], rhs.get().features[fx]);
  }

  bool operator()(const Example<Feature, Label>& lhs,
                  const Example<Feature, Label>& rhs) const {
    return cmp(lhs.features[fx], rhs.features[fx]);
  }

 private:
  Cmp cmp;
  std::size_t fx;
};

// A dataset is defined as a collection of training examples.
template <typename Feature, typename Label>
using DataSet = std::vector<Example<Feature, Label>>;

// A sampled dataset.  Examples are wrapped with a const reference wrapper to
// avoid expensive copies when sampling.  Note that this means that the dataset
// that provided the sample must outlive all sampled data sets.
template <typename Feature, typename Label>
using SampledDataSet = std::vector<SampledExample<Feature, Label>>;

// Generates an empty dataset with n_samples, each containing n_features.
template <typename Feature, typename Label,
          template <typename, typename> class DataSetT = DataSet>
DataSetT<Feature, Label> empty_data_set(std::size_t n_samples,
                                        std::size_t n_features) {
  DataSetT<Feature, Label> data_set(n_samples);
  for (auto& example : data_set) {
    example.features.resize(n_features);
  }
  return data_set;
}

// Sample n random items from the provided dataset with replacement.
template <typename Feature, typename Label>
SampledDataSet<Feature, Label> sample_with_replacement(
    const DataSet<Feature, Label>& data_set, std::size_t n) {
  SampledDataSet<Feature, Label> sample;
  for (std::size_t i = 0; i < n; ++i) {
    sample.push_back(data_set[random_range(0ul, data_set.size() - 1)]);
  }
  return sample;
}

}  // namespace rf
}  // namespace qp
#endif /* DATASET_H */
