#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <cassert>
#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "logging.h"

namespace qp {
namespace rf {

template <int NClusters, int Iterations>
class FeatureColumnKMeans {
 public:
  FeatureColumnKMeans() : projection_(NClusters, -1) {}

  void fit(const DataSet& data_set) {
    // Copy the data into a opencv matrix to make use of their functions.
    cv::Mat data(data_set.size(), data_set.front().features.size(), CV_32F);
    for (unsigned row = 0; row < data_set.size(); ++row) {
      for (unsigned col = 0; col < data_set[row].features.size(); ++col) {
        data.at<float>(row, col) = data_set[row].features[col];
      }
    }

    // Perform kmeans clustering on the transpose (i.e. cluster on feature
    // columns) of the matrix.
    cv::Mat labels;
    cv::kmeans(data.t(), NClusters, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                Iterations, 1.0),
               Iterations, cv::KMEANS_PP_CENTERS);
    assert(labels.rows == data_set.front().features.size());

    // Now this is tricky... we want to take 1 random feature column from each
    // cluster.  Naively, one could select random indicies until one of each
    // cluster label was found, but this method is technically
    // non-deterministic.  Instead, I used a variant of resevoir sampling.
    std::vector<std::size_t> seen_from_cluster(NClusters, 0);
    for (unsigned i = 0; i < labels.rows; ++i) {
      int cluster_id = labels.at<int>(i, 0);
      ++seen_from_cluster[cluster_id];
      // If we haven't seen any samples from this cluster before, take the
      // current feature.
      if (seen_from_cluster[cluster_id] == 1) {
        projection_[cluster_id] = i;
      } else {
        int seen = seen_from_cluster[cluster_id];
        if (random_real_range<double>(0, 1) <= (1.0 / seen)) {
          projection_[cluster_id] = i;
        }
      }
    }
  }

  void transform(DataSet& data_set) {
    for (auto& row : data_set) {
      row.features = project(row.features, projection_);
    }
  }

  std::vector<double> transform(const std::vector<double>& features) const {
    return project(features, projection_);
  }

  void fit_transform(DataSet& data_set) {
    fit(data_set);
    transform(data_set);
  }

 private:
  std::vector<FeatureIndex> projection_;
};

}  // namespace rf
}  // namespace qp

#endif /* CLUSTERING_H */
