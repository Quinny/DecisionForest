#ifndef MNIST_H
#define MNIST_H

#include <iostream>
#include "dataset.h"

namespace qp {
namespace rf {

// Read csv dataset from the given input stream.  Data should follow the format:
// label,f1,f2,f3...
DataSet read_csv_data_set(std::istream& is, std::size_t n_samples,
                          std::size_t n_features) {
  auto set = qp::rf::empty_data_set(n_samples, n_features);
  for (unsigned sample = 0; sample < n_samples; ++sample) {
    is >> set[sample].label;
    for (unsigned feature = 0; feature < n_features; ++feature) {
      // Ignore the comma.
      is.ignore(1);
      is >> set[sample].features[feature];
    }
  }

  return set;
}

}  // namespace rf
}  // namespace qp

#endif /* MNIST_H */
