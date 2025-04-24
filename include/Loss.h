#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>
#include <functional>

namespace NN {

enum class LossType { MSE = 0, CrossEntropy = 1 };

namespace LossFunctions {
double mseLossFunction(const Eigen::MatrixXd& predictions,
                       const Eigen::MatrixXd& targets);
Eigen::MatrixXd mseLossDerivative(const Eigen::MatrixXd& predictions,
                                  const Eigen::MatrixXd& targets);

double crossEntropyLossFunction(const Eigen::MatrixXd& predictions,
                                const Eigen::MatrixXd& targets);
Eigen::MatrixXd crossEntropyLossDerivative(const Eigen::MatrixXd& predictions,
                                           const Eigen::MatrixXd& targets);
}  // namespace LossFunctions

struct Loss {
  LossType type;
  std::function<double(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> loss;
  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>
      derivative;
};

Loss createLoss(LossType type);
}  // namespace NN

#endif  // LOSS_H