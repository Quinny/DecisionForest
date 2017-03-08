#ifndef SINGLE_LAYER_PERCEPTRON_H
#define SINGLE_LAYER_PERCEPTRON_H

#include <cassert>
#include <functional>
#include <numeric>
#include <vector>

#include "functional.h"
#include "random.h"
#include "vector_util.h"

namespace qp {
namespace rf {

namespace {

// Determine the amount to be added to the current weight.
double weight_delta(const double learning_rate, const double target,
                    const double output, const double feature) {
  return learning_rate * (target - output) * feature;
}

template <typename T>
using Matrix = std::vector<std::vector<T>>;

}  // namespace

// A single layer perceptron with a configurable activation function.
template <typename ActivationFn>
class SingleLayerPerceptron {
 public:
  // For testing only.
  SingleLayerPerceptron(const Matrix<double>& weights,
                        const std::vector<double>& biases, double learning_rate)
      : weights_(weights),
        biases_(biases),
        n_inputs_(weights.front().size()),
        n_outputs_(biases.size()),
        learning_rate_(learning_rate) {}

  // Initialize the layer with random weights and biases in [-1, 1].
  SingleLayerPerceptron(std::size_t n_inputs, std::size_t n_outputs,
                        double learning_rate)
      : n_inputs_(n_inputs),
        n_outputs_(n_outputs),
        learning_rate_(learning_rate) {
    double weight_range = 1 / std::sqrt(n_inputs);
    weights_.resize(n_outputs_);
    std::for_each(weights_.begin(), weights_.end(),
                  [this, weight_range](std::vector<double>& wv) {
                    generate_back_n(wv, n_inputs_,
                                    std::bind(random_real_range<double>,
                                              -weight_range, weight_range));
                  });

    generate_back_n(biases_, n_outputs_, []() { return 0; });
  }

  // Given a set of features, return the activation values of the output layers.
  std::vector<double> predict(const std::vector<double>& features) const {
    std::vector<double> output(n_outputs_);
    for (auto i = 0ul; i < n_outputs_; ++i) {
      output[i] =
          activate_(std::inner_product(weights_[i].begin(), weights_[i].end(),
                                       features.begin(), biases_[i]));
    }
    return output;
  }

  // Learn a training example and update the weights and biases accordingly.
  void learn(const std::vector<double>& features,
             const std::vector<double>& true_output) {
    const auto actual_output = predict(features);
    for (auto i = 0ul; i < n_outputs_; ++i) {
      for (auto weight = 0ul; weight < n_inputs_; ++weight) {
        weights_[i][weight] += weight_delta(learning_rate_, true_output[i],
                                            actual_output[i], features[weight]);
      }
      // Bias can be treated as a weight with a constant feature value of 1.
      biases_[i] +=
          weight_delta(learning_rate_, true_output[i], actual_output[i], 1);
    }
  }

  double maximum_activation() const { return activate_.max(); }

  double minimum_activation() const { return activate_.min(); }

  double fire_threshold() const { return activate_.mid(); }

 private:
  Matrix<double> weights_;      // n_outputs x n_inputs
  std::vector<double> biases_;  // 1 x n_outputs
  std::size_t n_inputs_;
  std::size_t n_outputs_;
  ActivationFn activate_;
  double learning_rate_;
};

// Simple step activation.
struct Step {
  double operator()(const double x) const { return x > 0 ? 1 : -1; }

  double max() const { return 1; }

  double mid() const { return 0; }

  double min() const { return -1; }
};

// Models the sigmoid function but is faster to compute, since abs is
// significantly cheaper than exp.
struct FastSigmoid {
  double operator()(const double x) const { return x / (1 + std::abs(x)); }

  double max() const { return 1; }

  double mid() const { return 0; }

  double min() const { return -1; }
};

// Sigmoid activation function.
struct Sigmoid {
  double operator()(const double x) const { return 1 / (1 + std::exp(-x)); }

  double max() const { return 1; }

  double mid() const { return 0.5; }

  double min() const { return 0; }
};

// Tanh activation function.
struct Tanh {
  double operator()(const double x) const { return std::tanh(x); }

  double max() const { return 1; }

  double mid() const { return 0; }

  double min() const { return -1; }
};

}  // namespace rf
}  // namespace qp

#endif /* SINGLE_LAYER_PERCEPTRON_H */
