#ifndef NN_BUILDER_H
#define NN_BUILDER_H

#include "NeuralNetwork.h"

namespace NN {

class Builder {
 private:
  NeuralNetwork* network;

 public:
  Builder();
  ~Builder();

  Builder& addLayer(int inputSize, int outputSize,
                    const Activation& activation);
  NeuralNetwork* build();
};

}  // namespace NN

#endif  // NN_BUILDER_H