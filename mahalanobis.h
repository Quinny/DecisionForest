#ifndef MAHALANOBIS_H
#define MAHALANOBIS_H

#include "dataset.h"

// TODO this includes a load of other headers, figure out which ones I actually
// need to improve compile times.
#include <cassert>
#include <opencv2/opencv.hpp>
#include <vector>

namespace qp {
namespace rf {

// Computes the mahalanobis distance from a point to a given distribution.
class MahalanobisCalculator {
 public:
  // Initialize with the distribution.
  void initialize(const cv::Mat& distribution) {
    cv::Mat convariance_matrix;
    cv::calcCovarMatrix(distribution, convariance_matrix, means_,
                        CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);
    cv::invert(convariance_matrix, inverse_convariance_matrix_, cv::DECOMP_SVD);
    init_ = true;
  }

  double distance(const cv::Mat& features) const {
    // TODO sanity check, remove later?
    assert(init_);
    return cv::Mahalanobis(features, means_, inverse_convariance_matrix_);
  }

 private:
  bool init_ = false;
  cv::Mat inverse_convariance_matrix_;
  cv::Mat means_;
};

}  // namespace rf
}  // namespace qp

#endif /* MAHALANOBIS_H */
