#include "../include/NeuralNetwork.h"

namespace NN {

NeuralNetwork::~NeuralNetwork() {
  for (auto layer : layers) delete layer;
}

void NeuralNetwork::addLayer(Layer *layer) { layers.push_back(layer); }

Eigen::MatrixXd NeuralNetwork::predict(const Eigen::MatrixXd &X) const {
  Eigen::MatrixXd output = X;
  Eigen::MatrixXd Z;
  for (auto layer : layers) output = layer->forward(output, Z);
  return output;
}

void NeuralNetwork::train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y,
                          int epochs, int batchSize, const Loss &lossFunction,
                          const OptimizerParams &optimizerParams) {
  int numSamples = X.cols();

  AdamOptimizer optimizer(layers.size(), optimizerParams.learningRate,
                          optimizerParams.beta1, optimizerParams.beta2,
                          optimizerParams.epsilon);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double totalLoss = 0;
    int totalCount = 0;

    for (int start = 0; start < numSamples; start += batchSize) {
      int bs = std::min(batchSize, numSamples - start);
      Eigen::MatrixXd batchX = X.middleCols(start, bs);
      Eigen::MatrixXd batchY = y.middleCols(start, bs);

      std::vector<Eigen::MatrixXd> activations{batchX};
      std::vector<Eigen::MatrixXd> Zs;

      for (auto layer : layers) {
        Eigen::MatrixXd Z;
        activations.push_back(layer->forward(activations.back(), Z));
        Zs.push_back(Z);
      }

      Eigen::MatrixXd output = activations.back();
      double batchLoss = lossFunction.loss(output, batchY);
      totalLoss += batchLoss * bs;
      totalCount += bs;

      Eigen::MatrixXd grad = lossFunction.derivative(output, batchY);
      for (int i = layers.size() - 1; i >= 0; --i) {
        Eigen::MatrixXd dW;
        Eigen::VectorXd db;
        grad = layers[i]->backward(activations[i], Zs[i], grad, dW, db);
        optimizer.update(i, layers[i]->getWeights(), layers[i]->getBiases(), dW,
                         db);
      }
    }

    std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / totalCount
              << "\n";
  }
}

TestInfo NeuralNetwork::evaluate(const Eigen::MatrixXd &X,
                                 const Eigen::MatrixXd &y) {
  Eigen::MatrixXd pred = predict(X);
  long correct = 0;
  for (int i = 0; i < pred.cols(); ++i) {
    Eigen::Index pi, yi;
    pred.col(i).maxCoeff(&pi);
    y.col(i).maxCoeff(&yi);
    if (pi == yi) ++correct;
  }
  return {correct, pred.cols()};
}

void NeuralNetwork::save(const std::string &filename) const {
  std::ofstream ofs(filename);

  assert(ofs && "Cannot open file for saving");
  assert(!layers.empty() && "No layers to save");

  ofs << layers.size() << std::endl;
  for (size_t i = 0; i < layers.size(); ++i) {
    const auto &layer = layers[i];

    assert(layer && "Layer pointer is null");

    const Eigen::MatrixXd &W = layer->getWeights();
    const Eigen::VectorXd &b = layer->getBiases();

    assert(W.rows() > 0 && W.cols() > 0 && "Invalid weight matrix size");
    assert(b.size() == W.rows() && "Bias vector size must match weight rows");

    ofs << static_cast<int>(layer->getActivation().type) << " " << W.rows()
        << " " << W.cols() << "\n";

    for (int r = 0; r < W.rows(); ++r) {
      for (int c = 0; c < W.cols(); ++c) {
        ofs << W(r, c) << " ";
      }
      ofs << "\n";
    }
    for (int idx = 0; idx < b.size(); ++idx) {
      ofs << b(idx) << " ";
    }
    ofs << "\n";
  }
}

void NeuralNetwork::load(const std::string &filename) {
  std::ifstream ifs(filename);

  assert(ifs && "Cannot open file for loading");

  for (auto l : layers) {
    delete l;
  }
  layers.clear();

  size_t numLayers = 0;
  ifs >> numLayers;

  assert(ifs && "Invalid format: cannot read layer count");
  assert(numLayers > 0 && "Number of layers must be positive");

  for (size_t i = 0; i < numLayers; ++i) {
    int typeInt = 0;
    int rows = 0, cols = 0;
    ifs >> typeInt >> rows >> cols;

    assert(ifs && "Invalid format at layer header");
    assert(rows > 0 && cols > 0 && "Invalid matrix dimensions");

    auto actType = static_cast<ActivationType>(typeInt);

    assert(actType >= ActivationType::Sigmoid &&
           actType <= ActivationType::Relu && "Unknown activation type");

    Eigen::MatrixXd W(rows, cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        ifs >> W(r, c);
        assert(ifs && "Invalid weight data");
      }
    }

    Eigen::VectorXd b(rows);
    for (int idx = 0; idx < rows; ++idx) {
      ifs >> b(idx);
      assert(ifs && "Invalid bias data");
    }

    Activation act = NN::createActivation(actType);
    layers.push_back(new Layer(W, b, act));
  }
}

}  // namespace NN