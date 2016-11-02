#include <fstream>
#include <iostream>

#include "csv.h"
#include "deep_forest.h"
#include "forest.h"
#include "logging.h"
#include "split_fns.h"

// TODO command line arguments.
int main() {
  std::ios_base::sync_with_stdio(false);

  std::ifstream training_stream("mnist_train.csv");
  std::ifstream testing_stream("mnist_test.csv");

  if (!training_stream || !testing_stream) {
    std::cerr << "failed to open one of the files" << std::endl;
    return 1;
  }

  qp::LOG << "reading data..." << std::endl;
  auto training =
      qp::rf::read_csv_data_set<int, int>(training_stream, 50000, 784);
  auto testing =
      qp::rf::read_csv_data_set<int, int>(testing_stream, 10000, 784);

  qp::rf::LayerConfig input{200, 5};
  std::vector<qp::rf::LayerConfig> hidden{};
  qp::rf::LayerConfig output{200, 15};

  qp::rf::DeepForest<int, int, qp::rf::PerceptronSplit<int, int, 2, 1>,
                     qp::rf::PerceptronSplit<double, int, 2, 5>>
      forest(input, hidden, output);

  // qp::rf::DecisionForest<int, int, qp::rf::PerceptronSplit<int, int, 1, 1>>
  //    forest(
  //        /* trees=*/200, /*max_depth=*/17);

  qp::LOG << "training..." << std::endl;
  forest.train(training);

  qp::LOG << "predicting..." << std::endl;
  int correctly_classified = 0;
  for (unsigned i = 0; i < testing.size(); ++i) {
    if (forest.predict(testing[i].features) == testing[i].label) {
      ++correctly_classified;
    }
  }

  std::cout << (correctly_classified / static_cast<double>(testing.size()))
            << std::endl;
}
