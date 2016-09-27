#include <fstream>
#include <iostream>

#include "csv.h"
#include "forest.h"
#include "split_fns.h"

// TODO command line arguments.
int main() {
  std::ifstream training_stream("../mnist/mnist_train.csv");
  std::ifstream testing_stream("../mnist/mnist_test.csv");

  std::cout << "reading data..." << std::endl;
  auto training =
      qp::rf::read_csv_data_set<int, int>(training_stream, 50000, 784);
  auto testing =
      qp::rf::read_csv_data_set<int, int>(testing_stream, 10000, 784);

  qp::rf::DecisionForest<int, int, qp::rf::PerceptronSplit<2>> forest(100, 15);

  // qp::rf::DecisionForest<int, int, qp::rf::NDimensionalSplit<1>> forest(100,
  //                                                                      15);
  std::cout << "training..." << std::endl;
  forest.train(training);

  std::cout << "predicting..." << std::endl;
  int correctly_classified = 0;
  for (unsigned i = 0; i < testing.size(); ++i) {
    if (forest.predict(testing[i].features) == testing[i].label) {
      ++correctly_classified;
    }
  }

  std::cout << (correctly_classified / static_cast<double>(testing.size()))
            << std::endl;
}
