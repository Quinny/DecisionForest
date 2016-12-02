#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "dataset.h"

#include <chrono>
#include <iostream>
#include <vector>

// Tools for benchmarking classifiers.

namespace qp {

namespace {

// Return how long it takes for a given function to run in seconds.
template <typename F>
double time_op(F&& f) {
  const auto t1 = std::chrono::high_resolution_clock::now();
  f();
  const auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
      .count();
}

}  // namespace

// All times are reported in seconds.
struct BenchmarkInfo {
  double training_time;
  double evaluation_time;
  double accuracy;
  std::vector<std::vector<int>> confusion_matrix;
};

// Make everything look pretty.
std::ostream& operator<<(std::ostream& os, const BenchmarkInfo& info) {
  os << "training time:\t" << info.training_time << std::endl
     << "evaluation time:\t" << info.evaluation_time << std::endl
     << "accuracy:\t" << info.accuracy << std::endl
     << "confusion matrix:" << std::endl;

  for (const auto& row : info.confusion_matrix) {
    for (const auto val : row) {
      os << val << "\t";
    }
    os << std::endl;
  }
  return os;
}

template <typename Classifier>
BenchmarkInfo benchmark(Classifier& classifier,
                        const rf::DataSet& training_data,
                        const rf::DataSet& testing_data, const int n_labels) {
  BenchmarkInfo ret;
  ret.training_time = time_op([&]() { classifier.train(training_data); });
  ret.evaluation_time = 0;
  std::vector<std::vector<int>> confusion_matrix(n_labels,
                                                 std::vector<int>(n_labels, 0));
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
