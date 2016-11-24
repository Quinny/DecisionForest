#include "node.h"
#include "dataset.h"
#include "gtest/gmock.h"
#include "gtest/gtest.h"

class NodeTest : public ::testing::Test {};

// Returns left if the first feature is greater than 0, and right otherwise.
struct ConstSplitter {
  void train(qp::rf::SDIter first, qp::rf::SDIter last) {}

  qp::rf::SplitDirection apply(const std::vector<double>& e) const {
    return e[0] > 0 ? qp::rf::SplitDirection::LEFT
                    : qp::rf::SplitDirection::RIGHT;
  }

  std::size_t n_input_features() const { return 1; }
};

TEST_F(NodeTest, Apply) {
  qp::rf::DecisionNode<ConstSplitter> node;

  auto data = qp::rf::empty_data_set(2, 2);
  data[0].features = {-1, 1};
  data[1].features = {1, 1};

  auto sampled = qp::rf::sample_exactly(data);
  node.train(sampled.begin(), sampled.end());

  EXPECT_EQ(node.split_direction(data[0].features),
            qp::rf::SplitDirection::RIGHT);
  EXPECT_EQ(node.split_direction(data[1].features),
            qp::rf::SplitDirection::LEFT);
}

TEST_F(NodeTest, Predict) {
  qp::rf::DecisionNode<ConstSplitter> node;

  auto data = qp::rf::empty_data_set(4, 2);
  data[0].label = 1;
  data[1].label = 1;
  data[2].label = 2;
  data[3].label = 3;

  auto sampled = qp::rf::sample_exactly(data);
  node.train(sampled.begin(), sampled.end());
  EXPECT_EQ(node.predict(), 1);
}
