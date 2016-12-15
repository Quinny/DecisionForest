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

  const auto means = qp::rf::zero_center_mean(training);
  qp::rf::zero_center_mean(testing, means);

  qp::LOG << "starting threadpool" << std::endl;
#ifndef N_WORKERS
  qp::threading::Threadpool thread_pool;
#else
  qp::threading::Threadpool thread_pool(N_WORKERS);
#endif

  qp::rf::DecisionForest<qp::rf::RandomUnivariateSplit> forest(10, -1, 1.0,
                                                               &thread_pool);

  qp::LOG << "evaluating classifier" << std::endl;
  const auto results = qp::benchmark(forest, training, testing);
  std::cout << results << std::endl;
}
