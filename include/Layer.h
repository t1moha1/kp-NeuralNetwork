//
// Created by Тимофей Тулинов on 26.03.2025.
//
#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <random>
#include "Activation.h"

namespace NN {
    class Layer {
        Eigen::MatrixXd weights;
        Eigen::VectorXd biases;
        NN::Activation activation;

    public:
        Layer(const Eigen::MatrixXd& weights, const Eigen::VectorXd& biases, NN::Activation activation);
        Layer(int input_size, int output_size, NN::Activation activation);

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, Eigen::MatrixXd& Z) const;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& gradOutput, double learningRate);
    };
}
#endif //LAYER_H
