//
// Created by Тимофей Тулинов on 26.03.2025.
//
#include "../include/Layer.h"
namespace NN {
    Layer::Layer(const Eigen::MatrixXd &weights, const Eigen::VectorXd &biases, NN::ActivationFunctions::Activation activation) :
    weights(weights), biases(biases), activation(std::move(activation)) {} //???

    Layer::Layer(const int input_size, const int output_size, NN::ActivationFunctions::Activation activation) : activation(std::move(activation)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        // Нормальное распределение с математическим ожиданием 0 и стандартным отклонением 1
        std::normal_distribution<> d(0, 1);

        double lower = -0.5;
        double upper = 0.5;
        double value;

        weights = Eigen::MatrixXd(output_size, input_size);
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                do {
                    value = d(gen);
                } while (value < lower || value > upper);
                weights(i, j) = value;
            }
        }

        biases = Eigen::VectorXd(output_size);
        for (int i = 0; i < output_size; ++i) {
            do {
                value = d(gen);
            } while (value < lower || value > upper);
            biases(i) = value;
        }
    }

    Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd& input, Eigen::MatrixXd& Z) const {
        Z = (weights * input);
        Z.colwise() += biases;
        return activation.func(Z);
    }

    Eigen::MatrixXd Layer::backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& gradOutput, double learningRate) {
        const int batchSize = input.cols();

        Eigen::MatrixXd dZ = gradOutput.array() * activation.derivative(Z).array();

        Eigen::MatrixXd dW = (dZ * input.transpose()) / batchSize;

        Eigen::VectorXd db = dZ.rowwise().sum() / batchSize;

        weights -= learningRate * dW;
        biases  -= learningRate * db;
        Eigen::MatrixXd dA_prev = weights.transpose() * dZ;
        return dA_prev;
    }
}





