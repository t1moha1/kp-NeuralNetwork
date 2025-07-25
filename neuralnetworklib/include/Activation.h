#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Eigen/Dense>
#include <functional>

namespace NN {
enum class ActivationType { Sigmoid = 0, Softmax = 1, Relu = 2 };

namespace ActivationFunctions {

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z);
Eigen::MatrixXd sigmoid_prime(const Eigen::MatrixXd& z);

Eigen::MatrixXd softmax(const Eigen::MatrixXd& z);
Eigen::MatrixXd softmax_prime(const Eigen::MatrixXd& z);

Eigen::MatrixXd relu(const Eigen::MatrixXd& z);
Eigen::MatrixXd relu_prime(const Eigen::MatrixXd& z);

}  // namespace ActivationFunctions

struct Activation {
  ActivationType type;
  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> func;
  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> derivative;
};

Activation createActivation(ActivationType type);

}  // namespace NN

#endif  // ACTIVATION_H