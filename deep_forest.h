#ifndef DEEP_FOREST_H
#define DEEP_FOREST_H

#include "dataset.h"
#include "forest.h"

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
             LayerConfig output_layer_config)
      : input_layer_(input_layer_config.trees, input_layer_config.depth),
        output_layer_(output_layer_config.trees, output_layer_config.depth) {
    for (const auto& config : hidden_layer_configs) {
      hidden_layers_.emplace_back(
          new DecisionForest<double, Label, Hack>(config.trees, config.depth));
    }
  }

  // Train the deep forest on the given dataset.
  void train(const DataSet<Feature, Label>& data_set) {
    std::cout << "input" << std::endl;
    input_layer_.train(data_set);
    auto transformed = input_layer_.transform(data_set);
    for (auto& layer : hidden_layers_) {
      std::cout << "hidden" << std::endl;
      layer->train(transformed);
      transformed = layer->transform(transformed);
    }

    std::cout << "output" << std::endl;
    output_layer_.train(transformed);
  }

  Label predict(const std::vector<Feature>& features) {
    auto transformed = input_layer_.transform(features);
    for (const auto& layer : hidden_layers_) {
      transformed = layer->transform(transformed);
    }
    return output_layer_.predict(transformed);
  }

 private:
  // TODO is there a way around this?  Compiler errors about stuff
  // not being moveable.
  using forest_ptr = std::unique_ptr<DecisionForest<double, Label, Hack>>;
  DecisionForest<Feature, Label, SplitterFn> input_layer_;
  std::vector<forest_ptr> hidden_layers_;
  DecisionForest<double, Label, Hack> output_layer_;
};

}  // namespace rf
}  // namespace qp

#endif /* DEEP_FOREST_H */
