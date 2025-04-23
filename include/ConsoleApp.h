#ifndef CONSOLE_APP_H
#define CONSOLE_APP_H

#include "NeuralNetwork.h"
#include "Builder.h"
#include "MNISTLoader.h"
#include "Activation.h"
#include "Loss.h"
#include <memory>

namespace NN {

    class ConsoleApp {
    public:
        void run();

    private:
        std::unique_ptr<NeuralNetwork> network;
        MNISTLoader trainLoader;
        MNISTLoader testLoader;
        int inputSize = 0;

        void showMenu();
        void createNetwork();
        void loadData();
        void trainNetwork();
        void testNetwork();
        void saveNetwork();
    };

} // namespace NN
