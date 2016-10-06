#ifndef MAHALANOBIS_H
#define MAHALANOBIS_H

#include "dataset.h"

// TODO this includes a load of other headers, figure out which ones I actually
// need to improve compile times.
#include <opencv2/opencv.hpp>
#include <vector>

namespace qp {
namespace rf {

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename Feature>
class MahalanobisCalculator {
 public:
  void initialize(const Matrix<Feature>& distribution) {
    Matrix<double> convariance_matrix;
    cv::calcCovarMatrix(distribution, convariance_matrix, means_,
                        CV_COVAR_NORMAL + CV_COVAR_ROWS);
    cv::invert(convariance_matrix, inverse_convariance_matrix_, cv::DECOMP_SVD);
  }

  double distance(const std::vector<Feature>& features) {
    return cv::Mahalanobis(features, means_, inverse_convariance_matrix_);
  }

 private:
  Matrix<double> inverse_convariance_matrix_;
  std::vector<double> means_;
};

}  // namespace rf
}  // namespace qp

#endif /* MAHALANOBIS_H */
