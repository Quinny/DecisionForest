#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "dataset.h"

#include <chrono>
#include <iostream>
#include <vector>

/*
 * This file contains utility functions for benchmarking performance and
 * accuracy of classifiers
 */

namespace qp {

namespace {

// Runs the function f and returns the execution time in seconds.
template <typename F>
double time_op(F&& f) {
  const auto t1 = std::chrono::high_resolution_clock::now();
  f();
  const auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
      .count();
}

}  // namespace

// A struct for storing classifier benchmarks.  All times are stored in
// seconds.
struct BenchmarkInfo {
  // Total time taken to train the model.
  double training_time;
  // Total time taken to evaluate the test set.
  double evaluation_time;
  // Number of correctly classified instances / total number of instances.
  double accuracy;
  // m[i][j] = the number of instances of class i, which were predicted to be
  // class j.
  std::vector<std::vector<int>> confusion_matrix;
};

// Pretty print the benchmark info.
std::ostream& operator<<(std::ostream& os, const BenchmarkInfo& info) {
  os << "training time:   " << info.training_time << std::endl
     << "evaluation time: " << info.evaluation_time << std::endl
     << "accuracy:        " << info.accuracy << std::endl
     << "confusion matrix:" << std::endl;

  for (const auto& row : info.confusion_matrix) {
    for (const auto val : row) {
      os << val << "\t";
    }
    os << std::endl;
  }
  return os;
}

// Run the benchmarks and return the info struct.  The classifer type should
// define train, and predict methods.
template <typename Classifier>
BenchmarkInfo benchmark(Classifier& classifier,
                        const rf::DataSet& training_data,
                        const rf::DataSet& testing_data) {
  BenchmarkInfo ret;
  ret.training_time = time_op([&]() { classifier.train(training_data); });
  ret.evaluation_time = 0;

  const auto max_label =
      std::max_element(training_data.begin(), training_data.end(),
                       [](const auto& lhs, const auto& rhs) {
                         return lhs.label < rhs.label;
                       })
          ->label;

  std::vector<std::vector<int>> confusion_matrix(
      max_label + 1, std::vector<int>(max_label + 1, 0));
  int correctly_classified = 0;

  for (const auto& example : testing_data) {
    double predicted;
    ret.evaluation_time +=
        time_op([&]() { predicted = classifier.predict(example.features); });
    if (predicted == example.label) {
      ++correctly_classified;
    }
    ++confusion_matrix[example.label][predicted];
  }

  ret.confusion_matrix = std::move(confusion_matrix);
  ret.accuracy =
      correctly_classified / static_cast<double>(testing_data.size());
  return ret;
}

}  // namespace qp

#endif /* BENCHMARK_H */
