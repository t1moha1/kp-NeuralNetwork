#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <Eigen/Dense>
#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"

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

    private:
        std::vector<Layer*> layers;
    };

}

#endif // NEURALNETWORK_H

