#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>
#include <vector>

namespace NN {
struct OptimizerParams {
  double learningRate;
  double beta1;
  double beta2;
  double epsilon;
};

class AdamOptimizer {
 public:
  AdamOptimizer(int num_layers, double learning_rate, double beta1 = 0.9,
                double beta2 = 0.999, double epsilon = 1e-8);

  void update(int layer_index, Eigen::MatrixXd& W, Eigen::VectorXd& b,
              const Eigen::MatrixXd& dW, const Eigen::VectorXd& db);

 private:
  double lr, b1, b2, eps;
  int t;
  std::vector<Eigen::MatrixXd> m_w, v_w;
  std::vector<Eigen::VectorXd> m_b, v_b;
};
}  // namespace NN

#endif
