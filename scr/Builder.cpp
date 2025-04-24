#include "../include/Builder.h"

namespace NN {

Builder::Builder() { network = new NeuralNetwork(); }

Builder::~Builder() = default;

Builder& Builder::addLayer(int inputSize, int outputSize,
                           const Activation& activation) {
  network->addLayer(new Layer(inputSize, outputSize, activation));
  return *this;
}

NeuralNetwork* Builder::build() { return network; }

}  // namespace NN
