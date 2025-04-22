#include "../include/NeuralNetwork.h"
#include <iostream>

namespace NN {

NeuralNetwork::~NeuralNetwork() {
    for (auto layer : layers)
        delete layer;
}

void NeuralNetwork::addLayer(Layer* layer) {
    layers.push_back(layer);
}

Eigen::MatrixXd NeuralNetwork::predict(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd output = X;
    Eigen::MatrixXd Z;
    for (auto layer : layers)
        output = layer->forward(output, Z);
    return output;
}

void NeuralNetwork::train(const Eigen::MatrixXd& X,
                          const Eigen::MatrixXd& y,
                          int epochs,
                          int batchSize,
                          double learningRate,
                          const Loss& lossFunction) {
    int numSamples = X.cols();
    AdamOptimizer optimizer(layers.size(), learningRate);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0;
        int totalCount = 0;

        for (int start = 0; start < numSamples; start += batchSize) {
            int bs = std::min(batchSize, numSamples - start);
            Eigen::MatrixXd batchX = X.middleCols(start, bs);
            Eigen::MatrixXd batchY = y.middleCols(start, bs);

            std::vector<Eigen::MatrixXd> activations{batchX};
            std::vector<Eigen::MatrixXd> Zs;

            for (auto layer : layers) {
                Eigen::MatrixXd Z;
                activations.push_back(layer->forward(activations.back(), Z));
                Zs.push_back(Z);
            }

            Eigen::MatrixXd output = activations.back();
            double batchLoss = lossFunction.loss(output, batchY);
            totalLoss += batchLoss * bs;
            totalCount += bs;

            Eigen::MatrixXd grad = lossFunction.derivative(output, batchY);
            for (int i = layers.size() - 1; i >= 0; --i) {
                Eigen::MatrixXd dW;
                Eigen::VectorXd db;
                grad = layers[i]->backward(activations[i], Zs[i], grad, dW, db);
                optimizer.update(i, layers[i]->getWeights(), layers[i]->getBiases(), dW, db);
            }
        }

        std::cout << "Epoch " << epoch + 1
                  << ", Loss: " << totalLoss / totalCount
                  << std::endl;
    }
}

double NeuralNetwork::evaluate(const Eigen::MatrixXd& X,
                               const Eigen::MatrixXd& y) {
    Eigen::MatrixXd pred = predict(X);
    int correct = 0;
    for (int i = 0; i < pred.cols(); ++i) {
        Eigen::Index pi, yi;
        pred.col(i).maxCoeff(&pi);
        y.col(i).maxCoeff(&yi);
        if (pi == yi)
            ++correct;
    }
    return static_cast<double>(correct) / pred.cols() * 100;
}

}