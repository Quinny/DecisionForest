#ifndef DEEP_FOREST_H
#define DEEP_FOREST_H

#include "dataset.h"
#include "forest.h"
#include "logging.h"

namespace qp {
namespace rf {

// A configuration for a layer of a deep forest.
struct LayerConfig {
  std::size_t trees;
  std::size_t depth;

  // For convenience sake. Allows for things like:
  // vector<LayerConfig> hidden_layers = {
  //    {50, 2}, {100, 5}, {200, 8}
  // };
  LayerConfig(std::initializer_list<std::size_t> init)
      : trees(*init.begin()), depth(*(init.begin() + 1)) {}
};

// A deep forest consists of layers of decision forests.  Each layer passes
// a transformed feature vector to the next.  The transformed feature vector
// consists of the mahalanobis distance from the given feature set to the
// distribution at each leaf node.
//
// TODO the template parameter "Hack" represents the splitter function for the
// hidden layers.  This is needed because the provided splitter function
// currently only works for the input layer.  This could be fixed with a simple
// type trait that subs in a different feature type to a given split function.
// More thought needs to be put into this...
template <typename Feature, typename Label, typename SplitterFn, typename Hack>
class DeepForest {
 public:
  // Construct the forest based on the layer configurations.
  DeepForest(LayerConfig input_layer_config,
             const std::vector<LayerConfig>& hidden_layer_configs,
             LayerConfig output_layer_config, double bag_percentage)
      : input_layer_(input_layer_config.trees, input_layer_config.depth,
                     bag_percentage),
        output_layer_(output_layer_config.trees, output_layer_config.depth,
                      bag_percentage) {
    hidden_layers_.reserve(hidden_layer_configs.size());
    for (const auto& config : hidden_layer_configs) {
      hidden_layers_.emplace_back(config.trees, config.depth, bag_percentage);
    }
  }

  // Train the deep forest on the given dataset.
  void train(const DataSet<Feature, Label>& data_set) {
    LOG << "training input layer" << std::endl;
    input_layer_.train(data_set);
    auto transformed = input_layer_.transform(data_set);
    for (auto& layer : hidden_layers_) {
      LOG << "training hidden layer" << std::endl;
      layer.train(transformed);
      LOG << "transforming data set" << std::endl;
      transformed = layer.transform(transformed);
    }

    LOG << "training output layer" << std::endl;
    output_layer_.train(transformed);
  }

  Label predict(const std::vector<Feature>& features) {
    auto transformed = input_layer_.transform(features);
    for (const auto& layer : hidden_layers_) {
      transformed = layer.transform(transformed);
    }
    return output_layer_.predict(transformed);
  }

 private:
  DecisionForest<Feature, Label, SplitterFn> input_layer_;
  std::vector<DecisionForest<double, Label, Hack>> hidden_layers_;
  DecisionForest<double, Label, Hack> output_layer_;
};

}  // namespace rf
}  // namespace qp

#endif /* DEEP_FOREST_H */
