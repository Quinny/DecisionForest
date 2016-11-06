#ifndef SINGLE_LAYER_PERCEPTRON_H
#define SINGLE_LAYER_PERCEPTRON_H

#include <cassert>
#include <functional>
#include <numeric>
#include <vector>
#include "functional.h"
#include "random.h"

namespace qp {
namespace rf {

namespace {

template <typename Feature>
double weight_update(const Feature& feature, const double error,
                     const double learning_rate, const double current_weight) {
  return current_weight + (learning_rate * error * feature);
}

template <typename T>
using Matrix = std::vector<std::vector<T>>;

}  // namespace

template <typename Feature, typename Label, typename ActivationFn>
class SingleLayerPerceptron {
 public:
  SingleLayerPerceptron(std::size_t n_inputs, std::size_t n_outputs,
                        double learning_rate)
      : n_inputs_(n_inputs),
        n_outputs_(n_outputs),
        learning_rate_(learning_rate) {
    weights_.resize(n_outputs_);
    std::for_each(
        weights_.begin(), weights_.end(), [this](std::vector<double>& wv) {
          generate_back_n(wv, n_inputs_,
                          std::bind(random_real_range<double>, -1, 1));
        });

    generate_back_n(biases_, n_outputs_,
                    std::bind(random_real_range<double>, -1, 1));
  }

  std::vector<double> predict(const std::vector<Feature>& features) const {
    std::vector<double> output(n_outputs_);
    for (auto i = 0ul; i < n_outputs_; ++i) {
      output[i] =
          activate_(std::inner_product(weights_[i].begin(), weights_[i].end(),
                                       features.begin(), -1 * biases_[i]));
    }
    return output;
  }

  void learn(const std::vector<Feature>& features,
             const std::vector<double>& true_output) {
    const auto actual_output = predict(features);
    for (auto i = 0ul; i < n_outputs_; ++i) {
      const auto error = true_output[i] - actual_output[i];
      for (auto weight = 0ul; weight < n_inputs_; ++weight) {
        weights_[i][weight] = weight_update(
            features[weight], error, learning_rate_, weights_[i][weight]);
      }
      // Bias can be treated as a weight with a constant feature value of 1.
      biases_[i] = weight_update(1, error, learning_rate_, biases_[i]);
    }
  }

 private:
  Matrix<double> weights_;      // n_outputs x n_inputs
  std::vector<double> biases_;  // 1 x n_outputs
  std::size_t n_inputs_;
  std::size_t n_outputs_;
  ActivationFn activate_;
  double learning_rate_;
};

struct StepActivation {
  double operator()(const double x) const { return x > 0 ? 1 : -1; }
};

}  // namespace rf
}  // namespace qp

#endif /* SINGLE_LAYER_PERCEPTRON_H */
