#ifndef NODE_H
#define NODE_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>

#include "criterion.h"
#include "dataset.h"
#include "functional.h"
#include "mahalanobis.h"

namespace qp {
namespace rf {

const int splits_to_try = 20;

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
      leaf_ = true;
      return;
    }

    double min_impurity = 1000;
    auto total_samples = static_cast<double>(end - start + 1);

    // Try different split functions and choose the one which results in the
    // least impurity.
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
        splitter_ = candidate_split;
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

  void initialize_mahalanobis(const SampledDataSet<Feature, Label>& dataset,
                              std::size_t start, std::size_t end) {
    x_ = random_range<std::size_t>(0, dataset.front().get().features.size());
    y_ = random_range<std::size_t>(0, dataset.front().get().features.size());

    std::size_t distro_size = 0;
    for (auto i = start; i != end; ++i) {
      if (dataset[i].get().label == prediction_) {
        ++distro_size;
      }
    }

    cv::Mat distribution(distro_size, 2, CV_64F);
    std::size_t c = 0;

    for (auto i = start; i != end; ++i) {
      if (dataset[i].get().label == prediction_) {
        distribution.at<double>(c, 0) = dataset[i].get().features[x_];
        distribution.at<double>(c, 1) = dataset[i].get().features[y_];
        ++c;
      }
    }

    mc_.initialize(distribution);
  }

  double mahalanobis_distance(const std::vector<Feature>& features) const {
    cv::Mat projected(1, 2, CV_64F);
    projected.at<double>(0, 0) = features[x_];
    projected.at<double>(0, 1) = features[y_];

    return mc_.distance(projected);
  }

  // Whether or not this node is ready to predict.
  bool leaf() const { return leaf_; }

 private:
  MahalanobisCalculator mc_;
  std::size_t x_;
  std::size_t y_;

  Label prediction_;
  SplitterFn splitter_;
  bool leaf_;
};

}  // namespace rf
}  // namespace qp

#endif /* NODE_H */
