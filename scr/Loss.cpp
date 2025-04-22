// Created by t1moha1 on 21.04.2025.

#include "../include/Loss.h"
#include <Eigen/Dense>

namespace NN {
    namespace LossFunctions {
        double mseLossFunction(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
            double n = static_cast<double>(predictions.size());
            return (predictions - targets).array().square().sum() / n;
        }

        Eigen::MatrixXd mseLossDerivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
            double n = static_cast<double>(predictions.size());
            return (predictions - targets) / n;
        }

        double crossEntropyLossFunction(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
            const double epsilon = 1e-12;
            Eigen::MatrixXd clipped = predictions.array().max(epsilon).min(1 - epsilon);
            int numExamples = predictions.cols();
            double lossSum = -(targets.array() * clipped.array().log()).sum();
            return lossSum / numExamples;
        }

        Eigen::MatrixXd crossEntropyLossDerivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
            int numExamples = predictions.cols();
            return (predictions - targets) / numExamples;
        }
    }

     Loss createLoss(LossType type) {
        switch (type) {
            case LossType::MSE:
                return {type,
                                  LossFunctions::mseLossFunction,
                                  LossFunctions::mseLossDerivative};
            case LossType::CrossEntropy:
                return {type,
                                LossFunctions::crossEntropyLossFunction,
                                LossFunctions::crossEntropyLossDerivative};
            default:
                return {LossType::MSE,
                                  LossFunctions::mseLossFunction,
                                  LossFunctions::mseLossDerivative};
        }
    }

}
