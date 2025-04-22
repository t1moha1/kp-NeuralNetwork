#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Eigen/Dense>
#include <functional>

namespace NN {
    namespace ActivationFunctions {

        Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z);
        Eigen::MatrixXd sigmoid_prime(const Eigen::MatrixXd& z);

        Eigen::MatrixXd softmax(const Eigen::MatrixXd& z);
        Eigen::MatrixXd softmax_prime(const Eigen::MatrixXd& z);

        Eigen::MatrixXd relu(const Eigen::MatrixXd& z);
        Eigen::MatrixXd relu_prime(const Eigen::MatrixXd& z);

    }

    struct Activation {
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> func;
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> derivative;
    };

}

#endif // ACTIVATION_H