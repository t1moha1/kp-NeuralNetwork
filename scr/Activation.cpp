
#include "../include/Activation.h"

namespace NN {
namespace ActivationFunctions {
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z) {
  return (1.0 / (1.0 + (-z.array()).exp())).matrix();
}

Eigen::MatrixXd sigmoid_prime(const Eigen::MatrixXd& z) {
  Eigen::MatrixXd s = sigmoid(z);
  return (s.array() * (1 - s.array())).matrix();
}

Eigen::MatrixXd softmax(const Eigen::MatrixXd& z) {
  Eigen::MatrixXd result(z.rows(), z.cols());
  for (int i = 0; i < z.cols(); ++i) {
    Eigen::VectorXd col = z.col(i);
    double maxVal = col.maxCoeff();
    Eigen::VectorXd expCol = (col.array() - maxVal).exp();
    double sumExp = expCol.sum();
    result.col(i) = expCol / sumExp;
  }
  return result;
}

Eigen::MatrixXd softmax_prime(const Eigen::MatrixXd& z) {
  Eigen::MatrixXd s = softmax(z);
  return (s.array() * (1 - s.array())).matrix();
}

Eigen::MatrixXd relu(const Eigen::MatrixXd& z) {
  return z.array().max(0.0).matrix();
}

Eigen::MatrixXd relu_prime(const Eigen::MatrixXd& z) {
  return (z.array() > 0.0).cast<double>().matrix();
}
}  // namespace ActivationFunctions

Activation createActivation(ActivationType type) {
  switch (type) {
    case ActivationType::Sigmoid:
      return {type, ActivationFunctions::sigmoid,
              ActivationFunctions::sigmoid_prime};
    case ActivationType::Softmax:
      return {type, ActivationFunctions::softmax,
              ActivationFunctions::softmax_prime};
    case ActivationType::Relu:
      return {type, ActivationFunctions::relu, ActivationFunctions::relu_prime};
    default:
      return {ActivationType::Sigmoid, ActivationFunctions::sigmoid,
              ActivationFunctions::sigmoid_prime};
  }
}
}  // namespace NN
