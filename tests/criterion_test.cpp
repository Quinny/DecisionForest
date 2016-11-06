#include "criterion.h"
#include "gtest/gtest.h"

class CriterionTest : public ::testing::Test {};

TEST_F(CriterionTest, ZeroImpurity) {
  std::map<int, std::size_t> label_histogram{{1, 10}};
  auto elements_impurity = qp::rf::gini_impurity(label_histogram);

  EXPECT_EQ(elements_impurity.first, 10);
  EXPECT_EQ(elements_impurity.second, 0);
}

TEST_F(CriterionTest, NonZeroImpurity) {
  std::map<int, std::size_t> label_histogram{{1, 3}, {4, 7}, {5, 2}};
  auto elements_impurity = qp::rf::gini_impurity(label_histogram);

  const double expected_size = 12;
  const double expected_impurity =
      ((3 / expected_size) * (1 - (3 / expected_size))) +
      ((7 / expected_size) * (1 - (7 / expected_size))) +
      ((2 / expected_size) * (1 - (2 / expected_size)));

  EXPECT_EQ(expected_size, elements_impurity.first);
  EXPECT_DOUBLE_EQ(expected_impurity, elements_impurity.second);
}
