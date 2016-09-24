#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

#include "dataset.h"
#include "forest.h"
#include "mnist.h"
#include "node.h"
#include "tree.h"

template <int N>
class NDimensionalSplit {
 public:
  void train(const qp::rf::SampledDataSet<int, int>& set, std::size_t s,
             std::size_t e) {
    for (int i = 0; i < N; ++i) {
      auto feature_index =
          qp::rf::random_range(0ul, set.front().get().features.size());
      auto feature_range = std::minmax_element(
          set.begin() + s, set.begin() + e,
          qp::rf::CompareOnFeature<int, int>(feature_index));
      auto low = feature_range.first->get().features[feature_index];
      auto high = feature_range.second->get().features[feature_index];
      auto threshold = qp::rf::random_range(low, high);

      feature_indexes.emplace_back(feature_index);
      thresholds.emplace_back(threshold);
    }
  }

  qp::rf::SplitDirection apply(const std::vector<int>& features) const {
    for (unsigned i = 0; i < feature_indexes.size(); ++i) {
      if (features[feature_indexes[i]] < thresholds[i]) {
        return qp::rf::SplitDirection::LEFT;
      }
    }
    return qp::rf::SplitDirection::RIGHT;
  }

 private:
  std::vector<int> feature_indexes;
  std::vector<int> thresholds;
};

int main() {
  std::ifstream training_stream("../mnist/mnist_train.csv");
  std::ifstream testing_stream("../mnist/mnist_test.csv");

  std::cout << "reading data..." << std::endl;
  auto training = qp::read_mnist_csv_data(training_stream, 50000);
  auto testing = qp::read_mnist_csv_data(testing_stream, 10000);
  std::cout << "done" << std::endl;

  std::cout << "creating forest..." << std::endl;
  qp::rf::DecisionForest<int, int, NDimensionalSplit<1>> forest(100, 15);
  std::cout << "training..." << std::endl;
  forest.train(training);

  int got = 0;
  std::cout << "predicting..." << std::endl;
  for (unsigned i = 0; i < testing.size(); ++i) {
    if (forest.predict(testing[i].features) == testing[i].label) {
      ++got;
    }
  }

  std::cout << got << std::endl;
}
