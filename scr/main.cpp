#include <iostream>
#include "../include/MNISTLoader.h"
#include "../include/Builder.h"


int main() {
    const std::string train_data_img_path = "../data/train-images.idx3-ubyte";
    const std::string train_data_label_path = "../data/train-labels.idx1-ubyte";
    const int train_size = 60000;
    const std::string test_data_img_path = "../data/t10k-images.idx3-ubyte";
    const std::string test_data_label_path = "../data/t10k-labels.idx1-ubyte";
    const int test_size = 10000;


    NN::MNISTLoader trainLoader;
    trainLoader.loadData(train_data_img_path, train_data_label_path, train_size);
    std::cout << "Обучающие данные загружены\n";

    NN::MNISTLoader testLoader;
    testLoader.loadData(test_data_img_path, test_data_label_path, test_size);
    std::cout << "Тестовые данные загружены\n";

    const NN::Activation sigmoidActivation = NN::createActivation(NN::ActivationType::Sigmoid);

    const NN::Activation softmaxActivation = NN::createActivation(NN::ActivationType::Softmax);

    const NN::Activation reluActivation = NN::createActivation(NN::ActivationType::Relu);

    const NN::Loss loss = NN::createLoss(NN::LossType::CrossEntropy);

    const NN::OptimizerParams optimizerParams {0.01, 0.9, 0.999, 1e-8};


     NN::Builder builder;
     builder.addLayer(784, 128, sigmoidActivation);
     builder.addLayer(128, 64, sigmoidActivation);
     builder.addLayer(64, 10, softmaxActivation);

     auto network = builder.build();

     network->train(trainLoader.images, trainLoader.labels, 25, 256, loss, optimizerParams);

    NN::TestInfo res = network->evaluate(testLoader.images, testLoader.labels);

     network->save("network_parameters_25.txt");

    delete network;

    //  auto net =  new NN::NeuralNetwork();
    //
    // net->load("network_parameters_25_5_5.txt");

    //net->train(trainLoader.images, trainLoader.labels, 15, 256, 0.0001, loss, optimizerParams);

    std::cout << static_cast<double>(res.correctPredictions) / res.totalTests * 100 << "\n";

    //net->save("network_parameters_25_5_5_15.txt");

    return 0;
}