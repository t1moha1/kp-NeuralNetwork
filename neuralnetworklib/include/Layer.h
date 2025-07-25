#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <random>

#include "Activation.h"

namespace NN {
class Layer {
 public:
  Eigen::MatrixXd& getWeights() { return weights; }
  Eigen::VectorXd& getBiases() { return biases; }
  const Activation& getActivation() const { return activation; }

  Layer(const Eigen::MatrixXd& weights, const Eigen::VectorXd& biases,
        Activation activation);
  Layer(int input_size, int output_size, Activation activation);

  Eigen::MatrixXd forward(const Eigen::MatrixXd& input,
                          Eigen::MatrixXd& Z) const;

  Eigen::MatrixXd backward(const Eigen::MatrixXd& input,
                           const Eigen::MatrixXd& Z,
                           const Eigen::MatrixXd& gradOutput,
                           Eigen::MatrixXd& dW, Eigen::VectorXd& db) const;

 private:
  Eigen::MatrixXd weights;
  Eigen::VectorXd biases;
  Activation activation;
};
}  // namespace NN

#endif  // LAYER_H
