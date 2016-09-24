#include "node.h"
#include <cassert>
#include "dataset.h"

struct ConstSplitter {
  void train(const qp::rf::DataSet<int, int>& set, int start, int end) {}

  qp::rf::SplitDirection apply(const std::vector<int>& e) const {
    return e[0] > 0 ? qp::rf::SplitDirection::LEFT
                    : qp::rf::SplitDirection::RIGHT;
  }
};

void node_apply_test() {
  qp::rf::DecisionNode<int, int, ConstSplitter> node;

  auto data = qp::rf::empty_data_set<int, int>(2, 2);
  data[0].features = {-1, 1};
  data[1].features = {1, 1};

  node.train(data, 0, 2);
  assert(node.split_direction(data[0].features) ==
         qp::rf::SplitDirection::RIGHT);
  assert(node.split_direction(data[1].features) ==
         qp::rf::SplitDirection::LEFT);
}

void node_predict_test() {
  qp::rf::DecisionNode<int, int, ConstSplitter> node;

  auto data = qp::rf::empty_data_set<int, int>(4, 2);
  data[0].label = 1;
  data[1].label = 1;
  data[2].label = 2;
  data[3].label = 3;

  node.train(data, 0, 4);
  assert(node.predict() == 1);
}

int main() {
  node_apply_test();
  node_predict_test();
}
