#ifndef MNIST_H
#define MNIST_H

#include <iostream>
#include "dataset.h"

namespace qp {

// Read mnist data from the given input stream.  Data should follow the
// following
// format:
// label, pix-11, pix-12, pix-13, ...
//
// A pair of (features, labels) will be returned.
rf::DataSet<int, int> read_mnist_csv_data(std::istream& is,
                                          std::size_t n_samples,
                                          std::size_t n_features = 784) {
  auto set = qp::rf::empty_data_set<int, int>(n_samples, n_features);
  for (int sample = 0; sample < n_samples; ++sample) {
    is >> set[sample].label;
    for (int feature = 0; feature < n_features; ++feature) {
      // Ignore the comma.
      is.ignore(1);
      is >> set[sample].features[feature];
    }
  }

  return set;
}

}  // namespace qp

#endif /* MNIST_H */
