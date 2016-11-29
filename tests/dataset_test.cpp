#include <sstream>

#include "csv.h"
#include "dataset.h"
#include "gtest/gmock.h"
#include "gtest/gtest.h"

using ::testing::DoubleEq;
using ::testing::ElementsAre;
class DataSetTest : public ::testing::Test {};

TEST_F(DataSetTest, CompareOnFeature) {
  auto dataset = qp::rf::empty_data_set(3, 3);
  dataset.front().features = {1, 2, 3};
  dataset[1].features = {5, 7, 9};
  dataset[2].features = {9, 0, 11};

  const auto t1 = std::max_element(dataset.begin(), dataset.end(),
                                   qp::rf::CompareOnFeature<>(1));
  EXPECT_THAT(t1->features, ElementsAre(5, 7, 9));

  const auto t2 = std::max_element(dataset.begin(), dataset.end(),
                                   qp::rf::CompareOnFeature<>(0));
  EXPECT_THAT(t2->features, ElementsAre(9, 0, 11));
}

TEST_F(DataSetTest, ModeLabel) {
  std::string csv_dataset =
      "1, 2, 3\n"
      "2, 7, 9\n"
      "1, 8, 7\n";

  std::stringstream stream(csv_dataset);
  const auto dataset = qp::rf::read_csv_data_set(stream, 3, 2);
  auto sampled = qp::rf::sample_exactly(dataset);

  const auto mode = qp::rf::mode_label(sampled.begin(), sampled.end());
  EXPECT_EQ(mode, 1);
}

TEST_F(DataSetTest, SingleLabel) {
  auto dataset = qp::rf::empty_data_set(3, 3);
  dataset[0].label = 1;
  dataset[1].label = 5;
  dataset[2].label = 9;

  auto sampled = qp::rf::sample_exactly(dataset);

  const auto t1 = qp::rf::single_label(sampled.begin(), sampled.end());
  EXPECT_FALSE(t1);

  dataset[1].label = 1;
  dataset[2].label = 1;

  const auto t2 = qp::rf::single_label(sampled.begin(), sampled.end());
  EXPECT_TRUE(t2);
}

TEST_F(DataSetTest, ZeroCenter) {
  std::string csv_dataset =
      "1, 2, 3\n"
      "2, 7, 9\n"
      "1, 8, 7\n";

  std::stringstream stream(csv_dataset);
  auto dataset = qp::rf::read_csv_data_set(stream, 3, 2);

  // Mean for f1 = 5.666666
  // Mean for f2 = 6.333333
  const auto means = qp::rf::zero_center_mean(dataset);

  EXPECT_THAT(means, ElementsAre(DoubleEq(5.666666), DoubleEq(6.33333)));

  EXPECT_THAT(dataset[0].features,
              ElementsAre(DoubleEq(-3.66667), DoubleEq(-3.33333)));
  EXPECT_THAT(dataset[1].features,
              ElementsAre(DoubleEq(1.33333), DoubleEq(2.66667)));
  EXPECT_THAT(dataset[2].features,
              ElementsAre(DoubleEq(2.33333), DoubleEq(0.66667)));
}
