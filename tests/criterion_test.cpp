#include "criterion.h"
#include "gtest/gtest.h"

class CriterionTest : public ::testing::Test {};

TEST_F(CriterionTest, ZeroImpurity) {
  std::map<int, std::size_t> label_histogram{{1, 10}};
  auto elements_impurity = qp::rf::gini_impurity(label_histogram);

  EXPECT_EQ(elements_impurity.first, 10);
  EXPECT_EQ(elements_impurity.second, 0);
}
