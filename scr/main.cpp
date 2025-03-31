#include <iostream>
#include "MNISTLoader.cpp"
#include "Builder.cpp"


int main() {
    Activation sigmoidActivation{sigmoid, softmax_prime};

    Activation softmaxActivation{softmax, softmax_prime};

    Loss loss{crossEntropyLossFunction, crossEntropyLossDerivative};

    MNISTLoader trainLoader;
    if (!trainLoader.loadData("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", 60000)) {
        std::cerr << "Ошибка загрузки обучающих данных" << std::endl;
        return -1;
    }
    std::cout << "Обучающие данные загружены\n";

    //Загрузка тестовых данных
    MNISTLoader testLoader;
    if (!testLoader.loadData("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte", 10000)) {
        std::cerr << "Ошибка загрузки тестовых данных" << std::endl;
        return -1;
    }
    std::cout << "Тестовые данные загружены\n";

    Builder builder;
    builder.addLayer(784, 128, sigmoidActivation);
    builder.addLayer(128, 64, sigmoidActivation);
    builder.addLayer(64, 10, softmaxActivation);

    auto network = builder.build();

    network->train(trainLoader.images, trainLoader.labels, 10, 10, 0.05, loss);

    std::cout << network->evaluate(testLoader.images, testLoader.labels);

    delete network;

    return 0;
}