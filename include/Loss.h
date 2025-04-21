#ifndef LOSS_H
#define LOSS_H

#include <functional>
#include <Eigen/Dense>

namespace NN {
    namespace LossFunctions {
        double mseLossFunction(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets);
        Eigen::MatrixXd mseLossDerivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets);

        double crossEntropyLossFunction(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets);
        Eigen::MatrixXd crossEntropyLossDerivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets);
    }

    struct Loss {
        std::function<double(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> loss;
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> derivative;
    };
}

#endif // LOSS_H