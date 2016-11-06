#include "functional.h"
#include "gtest/gmock.h"
#include "gtest/gtest.h"

using ::testing::ElementsAre;
class FunctionalTest : public ::testing::Test {};

TEST_F(FunctionalTest, CompareOnSecond) {
  std::vector<std::pair<int, int>> pairs{{-1, 7}, {105, -1}, {2, 3}};

  const auto max = std::max_element(pairs.begin(), pairs.end(),
                                    qp::rf::CompareOnSecond<int, int>());

  const auto min = std::min_element(pairs.begin(), pairs.end(),
                                    qp::rf::CompareOnSecond<int, int>());

  EXPECT_EQ(max->first, -1);
  EXPECT_EQ(max->second, 7);

  EXPECT_EQ(min->first, 105);
  EXPECT_EQ(min->second, -1);
}

TEST_F(FunctionalTest, GenerateBackN) {
  std::vector<int> v;
  const auto g = []() { return 5; };

  qp::rf::generate_back_n(v, 5, g);
  EXPECT_EQ(v.size(), 5);
  EXPECT_THAT(v, ElementsAre(5, 5, 5, 5, 5));
}

TEST_F(FunctionalTest, Projection) {
  std::vector<int> original{1, 2, 3, 4, 5};
  std::vector<std::size_t> projection{0, 3, 4};

  auto projected = qp::rf::project(original, projection);
  EXPECT_EQ(projected.size(), 3);
  EXPECT_THAT(projected, ElementsAre(1, 4, 5));

  std::vector<std::size_t> null_projection{};
  auto null = qp::rf::project(original, null_projection);
  EXPECT_TRUE(null.empty());
}
