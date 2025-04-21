#include <iostream>
#include "MNISTLoader.cpp"
#include "Builder.cpp"


int main() {
    const std::string train_data_img_path = "../data/train-images.idx3-ubyte";
    const std::string train_data_label_path = "../data/train-images.idx3-ubyte";
    const int train_size = 60000;
    const std::string test_data_img_path = "../data/t10k-images.idx3-ubyte";
    const std::string test_data_label_path = "../data/t10k-labels.idx1-ubyte";
    const int test_size = 10000;


    MNISTLoader trainLoader;
    if (!trainLoader.loadData(train_data_img_path, train_data_label_path, train_size)) {
        std::cerr << "Ошибка загрузки обучающих данных" << "\n";;
        return -1;
    }
    std::cout << "Обучающие данные загружены\n";

    //Загрузка тестовых данных
    MNISTLoader testLoader;
    if (!testLoader.loadData(test_data_img_path, test_data_label_path, test_size)) {
        std::cerr << "Ошибка загрузки тестовых данных" << "\n";
        return -1;
    }
    std::cout << "Тестовые данные загружены\n";

    const NN::ActivationFunctions::Activation sigmoidActivation{NN::ActivationFunctions::sigmoid,
        NN::ActivationFunctions::softmax_prime};

    const NN::ActivationFunctions::Activation softmaxActivation{NN::ActivationFunctions::softmax,
        NN::ActivationFunctions::softmax_prime};

    const NN::LossFunctions::Loss loss{NN::LossFunctions::crossEntropyLossFunction,
        NN::LossFunctions::crossEntropyLossDerivative};


    NN::Builder builder;
    builder.addLayer(784, 128, sigmoidActivation);
    builder.addLayer(128, 64, sigmoidActivation);
    builder.addLayer(64, 10, softmaxActivation);

    auto network = builder.build();

    network->train(trainLoader.images, trainLoader.labels, 10, 10, 0.05, loss);

    std::cout << network->evaluate(testLoader.images, testLoader.labels);

    delete network;
    return 0;
}