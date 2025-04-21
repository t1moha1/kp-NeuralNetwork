//
// Created by Тимофей Тулинов on 14.03.2025.

#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <Eigen/Dense>
#include <functional>

namespace NN {
    namespace ActivationFunctions {
        struct Activation {
            std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> func;
            std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> derivative;
        };


        inline Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z) {
            return (1.0 / (1.0 + (-z.array()).exp())).matrix();
        }

        inline Eigen::MatrixXd sigmoid_prime(const Eigen::MatrixXd& z) {
            Eigen::MatrixXd s = sigmoid(z);
            return (s.array() * (1 - s.array())).matrix();
        }

        inline Eigen::MatrixXd softmax(const Eigen::MatrixXd& z) {
            Eigen::MatrixXd result(z.rows(), z.cols());
            for (int i = 0; i < z.cols(); i++) {
                Eigen::VectorXd col = z.col(i);

                double maxVal = col.maxCoeff();
                Eigen::VectorXd expCol = (col.array() - maxVal).exp();
                double sumExp = expCol.sum();
                result.col(i) = expCol / sumExp;
            }
            return result;
        }

        inline Eigen::MatrixXd softmax_prime(const Eigen::MatrixXd& z) {
            Eigen::MatrixXd s = softmax(z);
            return (s.array() * (1 - s.array())).matrix();
        }
    }
}

#endif //ACTIVATION_H
