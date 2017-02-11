#ifndef REDUCERS_H
#define REDUCERS_H

#include <cassert>
#include <opencv2/flann/flann_base.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "logging.h"

namespace qp {
namespace rf {

template <int MaxClusters, int Iterations>
class FeatureColumnHierarchicalKMeans {
 public:
  void fit(const DataSet& data_set) {
    // Copy the data into a opencv matrix to make use of their functions.
    cv::Mat data(data_set.size(), data_set.front().features.size(), CV_32F);
    for (unsigned row = 0; row < data_set.size(); ++row) {
      for (unsigned col = 0; col < data_set[row].features.size(); ++col) {
        data.at<float>(row, col) = data_set[row].features[col];
      }
    }

    // Determine the optimal number of clusters using hierarchical clustering.
    cv::Mat centers(
        std::min<int>(MaxClusters, data_set.front().features.size()),
        data_set.front().features.size(), CV_32F);
    cvflann::KMeansIndexParams params(32, Iterations,
                                      cvflann::FLANN_CENTERS_KMEANSPP);

    // cvflann doesn't work with CV mat, which is awesome...
    cvflann::Matrix<float> samplesMatrix((float*)data.data, data.rows,
                                         data.cols);
    cvflann::Matrix<float> centersMatrix((float*)centers.data, centers.rows,
                                         centers.cols);
    const auto optimal_clusters =
        cvflann::hierarchicalClustering<cvflann::L2<float>>(
            samplesMatrix, centersMatrix, params);
    centers = centers.rowRange(cv::Range(0, optimal_clusters));
    projection_.resize(optimal_clusters);

    qp::LOG << "optimal clusters: " << optimal_clusters << std::endl;

    // Perform kmeans clustering on the transpose (i.e. cluster on feature
    // columns) of the matrix.
    cv::Mat labels;
    cv::kmeans(data.t(), optimal_clusters, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                Iterations, 0.01),
               Iterations, cv::KMEANS_PP_CENTERS);
    assert(labels.rows == data_set.front().features.size());

    // Now this is tricky... we want to take 1 random feature column from each
    // cluster.  Naively, one could select random indicies until one of each
    // cluster label was found, but this method is technically
    // non-deterministic.  Instead, I used a variant of resevoir sampling.
    std::vector<std::size_t> seen_from_cluster(optimal_clusters, 0);
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

#endif /* REDUCERS_H */
