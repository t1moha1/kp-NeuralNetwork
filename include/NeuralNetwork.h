#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <Eigen/Dense>
#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"
#include <iostream>
#include <fstream>

namespace NN {

    class NeuralNetwork {
    public:
        NeuralNetwork() = default;
        ~NeuralNetwork();

        void addLayer(Layer* layer);
        Eigen::MatrixXd predict(const Eigen::MatrixXd& X) const;

        void train(const Eigen::MatrixXd& X,
                   const Eigen::MatrixXd& y,
                   int epochs,
                   int batchSize,
                   double learningRate,
                   const Loss& lossFunction);

        double evaluate(const Eigen::MatrixXd& X,
                        const Eigen::MatrixXd& y);

        void save(const std::string& filename) const;
        void load(const std::string& filename);

    private:
        std::vector<Layer*> layers;
    };

}

#endif // NEURALNETWORK_H

