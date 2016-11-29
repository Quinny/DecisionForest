#include "single_layer_perceptron.h"
#include "gtest/gmock.h"
#include "gtest/gtest.h"

#include <iostream>

using ::testing::ElementsAre;
class SingleLayerPerceptronTest : public ::testing::Test {};

struct EchoActivation {
  double operator()(const double x) const { return x; }
};

TEST_F(SingleLayerPerceptronTest, NoTraining) {
  qp::rf::SingleLayerPerceptron<EchoActivation> slp({{1, 1}, {2, 1}, {3, 0}},
                                                    {0, 1, 2}, 1);

  const auto output = slp.predict({2, 3});

  EXPECT_EQ(output.size(), 3);
  EXPECT_THAT(output, ElementsAre(5, 8, 8));
}

TEST_F(SingleLayerPerceptronTest, WeightUpdate) {
  qp::rf::SingleLayerPerceptron<EchoActivation> slp({{1, 1}, {2, 1}, {3, 0}},
                                                    {0, 1, 2}, 1);

  // This should cause no weight updates.
  slp.learn({2, 3}, {5, 8, 8});

  {
    const auto output = slp.predict({2, 3});
    EXPECT_EQ(output.size(), 3);
    EXPECT_THAT(output, ElementsAre(5, 8, 8));
  }

  slp.learn({2, 3}, {4, 7, 7});
  // New weights should now be:
  // [[-1, -2],
  //  [0, -2],
  //  [1, -3]]
  // New biases should now be:
  // [-1, 0, 1]

  {
    const auto output = slp.predict({10, 8});
    EXPECT_EQ(output.size(), 3);
    EXPECT_THAT(output, ElementsAre(-27, -16, -13));
  }
}

TEST_F(SingleLayerPerceptronTest, LearnOR) {
  struct Step {
    double operator()(double x) const { return x >= 0 ? 1 : -1; }
  };

  qp::rf::SingleLayerPerceptron<Step> slp(2, 1, 0.01);

  double t = 1;
  double f = -1;
  using Example = std::pair<std::vector<double>, std::vector<double>>;
  std::vector<Example> or_data_set{
      {{f, f}, {f}}, {{f, t}, {t}}, {{t, f}, {t}}, {{t, t}, {t}}};

  bool had_error = true;
  while (had_error) {
    had_error = false;
    for (const auto& example : or_data_set) {
      slp.learn(example.first, example.second);
    }

    for (const auto& example : or_data_set) {
      if (slp.predict(example.first).front() != example.second.front()) {
        had_error = true;
        break;
      }
    }
  }

  EXPECT_FALSE(had_error);
}
