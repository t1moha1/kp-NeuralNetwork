#include "../include/ConsoleApp.h"

#include <iostream>
#include <limits>
#include <string>

namespace NN {

void ConsoleApp::run() {
  bool exitFlag = false;
  while (!exitFlag) {
    showMenu();
    int choice;
    std::cin >> choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    switch (choice) {
      case 1:
        createNetwork();
        break;
      case 2:
        loadNetwork();
        break;
      case 3:
        loadData();
        break;
      case 4:
        trainNetwork();
        break;
      case 5:
        testNetwork();
        break;
      case 6:
        saveNetwork();
        break;
      case 7:
        exitFlag = true;
        break;
      default:
        std::cout << "Invalid option\n";
    }
  }
}

void ConsoleApp::showMenu() const {
  std::cout << "\n=== Neural Network Console Application ===\n"
            << "1. Create Network\n"
            << "2. Load Network from File\n"
            << "3. Load MNIST Data\n"
            << "4. Train Network\n"
            << "5. Test Network\n"
            << "6. Save Network\n"
            << "7. Exit\n"
            << "Select an option: ";
}

void ConsoleApp::createNetwork() {
  std::cout << "Enter the number of layers: ";
  int numLayers;
  std::cin >> numLayers;

  Builder builder;
  std::cout << "Enter the input layer size: ";
  int inputSize;
  std::cin >> inputSize;

  int prevSize = inputSize;
  for (int i = 1; i <= numLayers; ++i) {
    std::cout << "Layer " << i << " - number of neurons: ";
    int neurons;
    std::cin >> neurons;

    std::cout
        << "Select activation function (0: Sigmoid, 1: Softmax, 2: ReLU): ";
    int actType;
    std::cin >> actType;
    Activation activation =
        createActivation(static_cast<ActivationType>(actType));

    builder.addLayer(prevSize, neurons, activation);
    prevSize = neurons;
  }

  network.reset(builder.build());
  std::cout << "Network created with " << numLayers << " layers.\n";
}

void ConsoleApp::loadNetwork() {
  std::cout << "Enter the path to the network parameters file: ";
  std::string filename;
  std::getline(std::cin, filename);

  network = std::make_unique<NeuralNetwork>();
  network->load(filename);
  std::cout << "\nNetwork loaded from " << filename << "\n";
}

void ConsoleApp::loadData() {
  const std::string defaultTrainImg = "../data/train-images.idx3-ubyte";
  const std::string defaultTrainLbl = "../data/train-labels.idx1-ubyte";
  const std::string defaultTestImg = "../data/t10k-images.idx3-ubyte";
  const std::string defaultTestLbl = "../data/t10k-labels.idx1-ubyte";

  std::cout << "Path to training images [" << defaultTrainImg << "]: ";
  std::string trainImg;
  std::getline(std::cin, trainImg);
  if (trainImg.empty()) trainImg = defaultTrainImg;

  std::cout << "Path to training labels [" << defaultTrainLbl << "]: ";
  std::string trainLbl;
  std::getline(std::cin, trainLbl);
  if (trainLbl.empty()) trainLbl = defaultTrainLbl;

  std::cout << "Number of training examples (0 = all) [0]: ";
  std::string tmp;
  std::getline(std::cin, tmp);
  size_t trainCount = tmp.empty() ? 0 : std::stoul(tmp);

  std::cout << "Path to test images [" << defaultTestImg << "]: ";
  std::string testImg;
  std::getline(std::cin, testImg);
  if (testImg.empty()) testImg = defaultTestImg;

  std::cout << "Path to test labels [" << defaultTestLbl << "]: ";
  std::string testLbl;
  std::getline(std::cin, testLbl);
  if (testLbl.empty()) testLbl = defaultTestLbl;

  std::cout << "Number of test examples (0 = all) [0]: ";
  std::getline(std::cin, tmp);
  size_t testCount = tmp.empty() ? 0 : std::stoul(tmp);

  trainLoader.loadData(trainImg, trainLbl, trainCount);
  std::cout << "\nTraining data loaded\n";

  testLoader.loadData(testImg, testLbl, testCount);
  std::cout << "Test data loaded\n";
}

void ConsoleApp::trainNetwork() {
  if (!network) {
    std::cout << "Please create or load a network first\n";
    return;
  }
  if (trainLoader.images.size() == 0) {
    std::cout << "Please load MNIST data first\n";
    return;
  }

  std::cout << "Enter the number of epochs: ";
  int epochs;
  std::cin >> epochs;

  std::cout << "Enter batch size (256): ";
  int batchSize;
  std::cin >> batchSize;

  std::cout << "Enter learning rate (0.01): ";
  double lr;
  std::cin >> lr;

  std::cout << "Enter beta1 (0.9): ";
  double beta1;
  std::cin >> beta1;

  std::cout << "Enter beta2 (0.999): ";
  double beta2;
  std::cin >> beta2;

  std::cout << "Enter epsilon (1e-8): ";
  double epsilon;
  std::cin >> epsilon;

  std::cout << "Select loss function (0: MSE, 1: CrossEntropy): ";
  int lossType;
  std::cin >> lossType;

  OptimizerParams params{lr, beta1, beta2, epsilon};
  Loss loss = createLoss(static_cast<LossType>(lossType));

  std::cout << "---------------------------------------------------------\n";
  std::cout << "Training started:\n";
  std::cout << "Training may take some time\n";
  network->train(trainLoader.images, trainLoader.labels, epochs, batchSize,
                 loss, params);

  std::cout << "\n" << epochs << " epochs completed\n";
  std::cout << "---------------------------------------------------------\n";
}

void ConsoleApp::testNetwork() const {
  if (!network) {
    std::cout << "Please create or load a network first\n";
    return;
  }
  if (testLoader.images.size() == 0) {
    std::cout << "Please load MNIST data first\n";
    return;
  }

  TestInfo result = network->evaluate(testLoader.images, testLoader.labels);
  double accuracy = static_cast<double>(result.correctPredictions) /
                    result.totalTests * 100.0;

  std::cout << "---------------------------------------------------------\n";
  std::cout << "Testing network on MNIST data\n";
  std::cout << "Correct predictions: " << result.correctPredictions << "/"
            << result.totalTests << "\n";
  std::cout << "Accuracy: " << accuracy << "%\n";
  std::cout << "---------------------------------------------------------\n";
}

void ConsoleApp::saveNetwork() const {
  if (!network) {
    std::cout << "Please create or load a network first\n";
    return;
  }

  std::cout << "Enter path to save the network: ";
  std::string filename;
  std::cin >> filename;

  network->save(filename);
  std::cout << "\nNetwork saved: " << filename << "\n";
}

}  // namespace NN