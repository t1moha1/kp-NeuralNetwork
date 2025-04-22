#include "../include/Layer.h"

namespace NN {

    Layer::Layer(const Eigen::MatrixXd& weights,
                 const Eigen::VectorXd& biases,
                 Activation activation)
        : weights(weights), biases(biases), activation(std::move(activation)) {}

    Layer::Layer(int input_size,
                 int output_size,
                 Activation activation)
        : activation(std::move(activation)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0, 1);

        weights = Eigen::MatrixXd(output_size, input_size);
        for (int i = 0; i < output_size; ++i)
            for (int j = 0; j < input_size; ++j)
                weights(i, j) = dist(gen);

        biases = Eigen::VectorXd(output_size);
        for (int i = 0; i < output_size; ++i)
            biases(i) = dist(gen);
    }

    Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd& input,
                                    Eigen::MatrixXd& Z) const {
        Z = weights * input;
        Z.colwise() += biases;
        return activation.func(Z);
    }

    Eigen::MatrixXd Layer::backward(const Eigen::MatrixXd& input,
                                     const Eigen::MatrixXd& Z,
                                     const Eigen::MatrixXd& gradOutput,
                                     Eigen::MatrixXd& dW,
                                     Eigen::VectorXd& db) const {
        int batchSize = input.cols();
        Eigen::MatrixXd dZ = gradOutput.array() * activation.derivative(Z).array();
        dW = (dZ * input.transpose()) / batchSize;
        db = dZ.rowwise().sum() / batchSize;
        return weights.transpose() * dZ;
    }

} // namespace NN