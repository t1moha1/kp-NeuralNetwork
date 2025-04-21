//
// Created by Тимофей Тулинов on 28.03.2025.
//
#include "NeuralNetwork.cpp"
namespace NN {
    class Builder {
        NeuralNetwork* network;
    public:
        Builder() {
            network = new NeuralNetwork();
        }
        ~Builder() = default;

        Builder& addLayer(int inputSize, int outputSize, const NN::ActivationFunctions::Activation& activation) {
            network->addLayer(new Layer(inputSize, outputSize, activation));
            return *this;
        }
        NeuralNetwork* build() {
            return network;
        }
    };
}
