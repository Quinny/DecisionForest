#ifndef DEEP_FOREST_H
#define DEEP_FOREST_H

#include "dataset.h"
#include "forest.h"
#include "logging.h"

/*
 * A DeepForest is analogous to a deep network, where the layers of the
 * network are random forests.
 */

namespace qp {
namespace rf {

// A configuration for a layer of a deep forest.
struct LayerConfig {
  // The number of trees in the layer.
  std::size_t trees;

  // The depth limit of the layer.
  int depth;

  // For convenience sake. Allows for things like:
  // vector<LayerConfig> hidden_layers = {
  //    {50, 2}, {100, 5}, {200, 8}
  // };
  LayerConfig(std::initializer_list<int> init)
      : trees(*init.begin()), depth(*(init.begin() + 1)) {}
};

// A deep forest consists of layers of decision forests.  Each layer passes
// a transformed feature vector to the next.
template <typename SplitterFn>
class DeepForest {
 public:
  // Construct the forest based on the layer configurations.
  DeepForest(const LayerConfig& input_layer_config,
             const std::vector<LayerConfig>& hidden_layer_configs,
             const LayerConfig& output_layer_config,
             qp::threading::Threadpool* thread_pool)
      : input_layer_(input_layer_config.trees, input_layer_config.depth,
                     thread_pool, TreeType::DEEP_FOREST),
        // The output layer can act as a single forest, as it performs no
        // transformations.
        output_layer_(output_layer_config.trees, output_layer_config.depth,
                      thread_pool, TreeType::SINGLE_FOREST) {
    hidden_layers_.reserve(hidden_layer_configs.size());
    for (const auto& config : hidden_layer_configs) {
      hidden_layers_.emplace_back(config.trees, config.depth, thread_pool,
                                  TreeType::DEEP_FOREST);
    }
  }

  // Train the deep forest on the given dataset.
  void train(const DataSet& data_set) {
    LOG << "training input layer" << std::endl;
    input_layer_.train(data_set);
    LOG << "transforming input layer" << std::endl;
    input_layer_.transform(data_set);

    for (auto& layer : hidden_layers_) {
      LOG << "training hidden layer" << std::endl;
      layer.train(data_set);
      LOG << "transforming data set" << std::endl;
      layer.transform(data_set);
    }

    LOG << "training output layer" << std::endl;
    output_layer_.train(data_set);
  }

  // Predict the label of a given feature set.
  double predict(const std::vector<double>& features) {
    auto transformed = input_layer_.transform(features);
    for (unsigned i = 0; i < hidden_layers_.size(); ++i) {
      transformed = hidden_layers_[i].transform(transformed);
    }
    return output_layer_.predict(transformed);
  }

 private:
  DecisionForest<SplitterFn> input_layer_;
  std::vector<DecisionForest<SplitterFn>> hidden_layers_;
  DecisionForest<SplitterFn> output_layer_;
};

}  // namespace rf
}  // namespace qp

#endif /* DEEP_FOREST_H */
