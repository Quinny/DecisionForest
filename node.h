#ifndef NODE_H
#define NODE_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "criterion.h"
#include "dataset.h"
#include "functional.h"
#include "mahalanobis.h"

namespace qp {
namespace rf {

enum class SplitDirection { LEFT, RIGHT };

// Represents a single node in a tree.
// TODO Only decide on a subset of features.
template <typename Feature, typename Label, typename SplitterFn>
class DecisionNode {
 public:
  DecisionNode() : leaf_(false){};

  // Train this node to decide on the dataset rows between start and end.
  void train(const SampledDataSet<Feature, Label>& dataset, std::size_t start,
             std::size_t end) {
    // The prediction at this node is the most occuring label in the incoming
    // samples.
    prediction_ = mode_label<Feature, Label>(dataset.begin() + start,
                                             dataset.begin() + end);

    // If the dataset only contains one label, then there is no point in
    // training this node, we can predict early.
    if (single_label<Feature, Label>(dataset.begin() + start,
                                     dataset.begin() + end)) {
      make_leaf();
    }

    double min_impurity = std::numeric_limits<double>::max();
    auto total_samples = static_cast<double>(end - start + 1);

    // Try different split functions and choose the one which results in the
    // least impurity.

    const auto total_features = dataset.front().get().features.size();
    const auto splits_to_try =
        std::sqrt(total_features) * splitter_.n_input_features();
    for (int i = 0; i < splits_to_try; ++i) {
      SplitterFn candidate_split;
      candidate_split.train(dataset, start, end);

      std::map<Label, std::size_t> went_left, went_right;
      for (unsigned sample = start; sample != end; ++sample) {
        if (candidate_split.apply(dataset[sample].get().features) ==
            SplitDirection::LEFT) {
          ++went_left[dataset[sample].get().label];
        } else {
          ++went_right[dataset[sample].get().label];
        }
      }

      auto left_impurity = gini_impurity(went_left);
      auto right_impurity = gini_impurity(went_right);
      auto total_impurity =
          (left_impurity.first / total_samples) * left_impurity.second +
          (right_impurity.first / total_samples) * right_impurity.second;

      if (total_impurity < min_impurity) {
        min_impurity = total_impurity;
        splitter_ = std::move(candidate_split);
      }
    }
  }

  // Determine the direction of the split based on the feautres.
  SplitDirection split_direction(const std::vector<Feature>& features) const {
    return splitter_.apply(features);
  }

  // Predict the label at this node based on the mode label of the incoming
  // samples.
  Label predict() const { return prediction_; }

  // Initialize the mahalanobis distance calculator.  Picks two random features
  // and then calcuates the distribution of those features for the dominant
  // label at this node.
  void initialize_mahalanobis(const SampledDataSet<Feature, Label>& dataset,
                              std::size_t start, std::size_t end) {
    distro_project_ = splitter_.get_features();

    std::size_t distro_size = 0;
    for (auto i = start; i != end; ++i) {
      if (dataset[i].get().label == prediction_) {
        ++distro_size;
      }
    }

    cv::Mat distribution(distro_size, distro_project_.size(), CV_64F);
    std::size_t c = 0;

    for (auto i = start; i != end; ++i) {
      if (dataset[i].get().label == prediction_) {
        for (int j = 0; j < distro_project_.size(); ++j) {
          distribution.at<double>(c, j) =
              dataset[i].get().features[distro_project_[j]];
        }
        ++c;
      }
    }

    mc_.initialize(distribution);
  }

  double mahalanobis_distance(const std::vector<Feature>& features) const {
    cv::Mat projected(1, distro_project_.size(), CV_64F);
    for (int i = 0; i < distro_project_.size(); ++i) {
      projected.at<double>(0, i) = features[distro_project_[i]];
    }

    return mc_.distance(projected);
  }

  // Whether or not this node is ready to predict.
  bool leaf() const { return leaf_; }

  bool make_leaf() {
    leaf_ = true;
    left_.reset();
    right_.reset();
  }

  DecisionNode<Feature, Label, SplitterFn>* make_child(SplitDirection dir) {
    if (dir == SplitDirection::LEFT) {
      left_.reset(new DecisionNode<Feature, Label, SplitterFn>());
      return left_.get();
    } else {
      right_.reset(new DecisionNode<Feature, Label, SplitterFn>());
      return right_.get();
    }
  }

  const DecisionNode<Feature, Label, SplitterFn>* get_child(
      SplitDirection dir) const {
    return dir == SplitDirection::LEFT ? left_.get() : right_.get();
  }

 private:
  MahalanobisCalculator mc_;
  std::vector<FeatureIndex> distro_project_;

  std::unique_ptr<DecisionNode<Feature, Label, SplitterFn>> left_, right_;

  Label prediction_;
  SplitterFn splitter_;
  bool leaf_;
};

}  // namespace rf
}  // namespace qp

#endif /* NODE_H */
