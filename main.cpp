#include <fstream>
#include <iostream>

#include "benchmark.h"
#include "csv.h"
#include "deep_forest.h"
#include "forest.h"
#include "logging.h"
#include "split_fns.h"
#include "threadpool.h"

int main() {
  std::ios_base::sync_with_stdio(false);

  std::ifstream training_stream("mnist_train.csv");
  std::ifstream testing_stream("mnist_test.csv");

  if (!training_stream || !testing_stream) {
    std::cerr << "failed to open one of the files" << std::endl;
    return 1;
  }

  qp::LOG << "reading data" << std::endl;
  auto training = qp::rf::read_csv_data_set(training_stream, 60000, 784);
  auto testing = qp::rf::read_csv_data_set(testing_stream, 10000, 784);

  // Subtract mean.
  const auto means = qp::rf::zero_center_mean(training);
  qp::rf::zero_center_mean(testing, means);

  qp::LOG << "starting threadpool" << std::endl;
#ifndef N_WORKERS
  qp::threading::Threadpool thread_pool;
#else
  qp::threading::Threadpool thread_pool(N_WORKERS);
#endif

  qp::LOG << "evaluating classifier" << std::endl;

  // Create a classic random univariate forest which will be fully grown.
  // This template parameter can be replaced with any of those defined in
  // split_fns.h to create different forests.
  qp::rf::DecisionForest<qp::rf::RandomUnivariateSplit> forest(10, -1,
                                                               &thread_pool);

  const auto results = qp::benchmark(forest, training, testing);
  std::cout << results << std::endl;
}
