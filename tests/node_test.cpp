#include "node.h"
#include "dataset.h"
#include "gtest/gmock.h"
#include "gtest/gtest.h"

class NodeTest : public ::testing::Test {};

// Returns left if the first feature is greater than 0, and right otherwise.
struct ConstSplitter {
  void train(const qp::rf::SampledDataSet<int, int>& set, int start, int end) {}

  qp::rf::SplitDirection apply(const std::vector<int>& e) const {
    return e[0] > 0 ? qp::rf::SplitDirection::LEFT
                    : qp::rf::SplitDirection::RIGHT;
  }
};

TEST_F(NodeTest, Apply) {
  qp::rf::DecisionNode<int, int, ConstSplitter> node;

  auto data = qp::rf::empty_data_set<int, int>(2, 2);
  data[0].features = {-1, 1};
  data[1].features = {1, 1};

  const auto sampled = qp::rf::sample_exactly(data);
  node.train(sampled, 0, 2);

  EXPECT_EQ(node.split_direction(data[0].features),
            qp::rf::SplitDirection::RIGHT);
  EXPECT_EQ(node.split_direction(data[1].features),
            qp::rf::SplitDirection::LEFT);
}

TEST_F(NodeTest, Predict) {
  qp::rf::DecisionNode<int, int, ConstSplitter> node;

  auto data = qp::rf::empty_data_set<int, int>(4, 2);
  data[0].label = 1;
  data[1].label = 1;
  data[2].label = 2;
  data[3].label = 3;

  const auto sampled = qp::rf::sample_exactly(data);
  node.train(sampled, 0, 4);
  EXPECT_EQ(node.predict(), 1);
}
