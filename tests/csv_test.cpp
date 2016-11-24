#include <sstream>

#include "csv.h"
#include "gtest/gmock.h"
#include "gtest/gtest.h"

using ::testing::ElementsAre;
class CsvTest : public ::testing::Test {};

TEST_F(CsvTest, ReadAll) {
  std::string csv_dataset =
      "1, 2, 3, 4\n"
      "2, 7, 8, 6\n";

  std::stringstream stream(csv_dataset);

  const auto dataset = qp::rf::read_csv_data_set(stream, /* n_samples=*/2,
                                                 /*n_features=*/3);

  EXPECT_EQ(dataset.size(), 2);

  EXPECT_EQ(dataset.front().label, 1);
  EXPECT_THAT(dataset.front().features, ElementsAre(2, 3, 4));

  EXPECT_EQ(dataset[1].label, 2);
  EXPECT_THAT(dataset[1].features, ElementsAre(7, 8, 6));
}

TEST_F(CsvTest, ReadSome) {
  std::string csv_dataset =
      "1, 2, 0.3, 4.7\n"
      "2, 7, 8, 6\n"
      "1, 0.7, 8, 6\n";

  std::stringstream stream(csv_dataset);

  const auto dataset = qp::rf::read_csv_data_set(stream, /* n_samples=*/2,
                                                 /*n_features=*/3);

  EXPECT_EQ(dataset.size(), 2);

  EXPECT_EQ(dataset.front().label, 1);
  EXPECT_THAT(dataset.front().features, ElementsAre(2, 0.3, 4.7));

  EXPECT_EQ(dataset[1].label, 2);
  EXPECT_THAT(dataset[1].features, ElementsAre(7, 8, 6));
}
