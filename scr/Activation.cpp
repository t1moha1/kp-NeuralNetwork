
#include "../include/Activation.h"
#include <Eigen/Dense>

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
    }
}
