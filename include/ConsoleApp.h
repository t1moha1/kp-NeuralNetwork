#ifndef CONSOLE_APP_H
#define CONSOLE_APP_H

#include <memory>
#include <string>

#include "Activation.h"
#include "Builder.h"
#include "Loss.h"
#include "MNISTLoader.h"
#include "NeuralNetwork.h"
#include "Optimizer.h"

namespace NN {

class ConsoleApp {
 public:
  void run();

 private:
  std::unique_ptr<NeuralNetwork> network;
  MNISTLoader trainLoader;
  MNISTLoader testLoader;

  void showMenu() const;
  void createNetwork();
  void loadNetwork();
  void loadData();
  void trainNetwork();
  void testNetwork() const;
  void saveNetwork() const;
};

}  // namespace NN

#endif  // CONSOLE_APP_H
