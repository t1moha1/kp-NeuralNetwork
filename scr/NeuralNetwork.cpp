//
// Created by Тимофей Тулинов on 28.03.2025.
//
#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include "../include/Layer.h"
#include "../include/Loss.h"


class NeuralNetwork {
public:
    std::vector<Layer*> layers;

    ~NeuralNetwork() {
        for (const Layer* layer : layers)
            delete layer;
    }

    void addLayer(Layer* layer) {
        layers.push_back(layer);
    }

    // Метод predict:
    // Последовательно выполняет прямой проход через все слои.
    // X имеет размер (inputSize, numSamples).
   Eigen:: MatrixXd predict(const Eigen::MatrixXd& X) const {
        Eigen::MatrixXd output = X;
        Eigen::MatrixXd Z;
        for (const Layer* layer : layers) {
            output = layer->forward(output, Z);
        }
        return output;
    }

    // Метод train:
    // Обучает сеть с использованием батчей, MSE и градиентного спуска.
    // Здесь входные данные X имеют размер (inputSize, numSamples), а y – (outputSize, numSamples).
    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
               int epochs, int batchSize, double learningRate, const Loss& lossFunction) {
        int numSamples = X.cols(); // число образцов = число столбцов
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            int totalExamples = 0;
            // Перебор батчей по столбцам
            for (int i = 0; i < numSamples; i += batchSize) {
                int currentBatchSize = std::min(batchSize, numSamples - i);
                // Извлекаем батч: каждый батч — набор столбцов
                Eigen::MatrixXd batchX = X.middleCols(i, currentBatchSize);
                Eigen::MatrixXd batchY = y.middleCols(i, currentBatchSize);

                // Прямой проход: сохраняем входы (A) и линейные комбинации (Z) для каждого слоя
                std::vector<Eigen::MatrixXd> A;  // A[0] – вход, A[l] – выход l-го слоя
                std::vector<Eigen::MatrixXd> Zs; // Zs[l] – значение Z в слое l
                A.push_back(batchX);
                for (const Layer* layer : layers) {
                    Eigen::MatrixXd Z;
                    Eigen::MatrixXd A_next = layer->forward(A.back(), Z);
                    Zs.push_back(Z);
                    A.push_back(A_next);
                }

                Eigen::MatrixXd output = A.back();
                // Вычисляем потерю для текущего батча и суммируем с учетом количества примеров в батче
                double batchLoss = lossFunction.loss(output, batchY);
                totalLoss += batchLoss * currentBatchSize;
                totalExamples += currentBatchSize;

                // Вычисляем градиент, используя выбранную функцию производной
                Eigen::MatrixXd dA = lossFunction.derivative(output, batchY);
                // Обратное распространение: начинаем с последнего слоя
                Eigen::MatrixXd grad = dA;
                for (int l = layers.size() - 1; l >= 0; l--) {
                    grad = layers[l]->backward(A[l], Zs[l], grad, learningRate);
                }
            }
            double averageLoss = totalLoss / totalExamples;
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << averageLoss << "\n";
        }
    }
    // Новый метод evaluate:
    // Принимает тестовые данные X и y, вычисляет предсказания и возвращает процент правильных ответов.
    // Для каждого столбца (примера) выбирается индекс максимального значения как предсказанная метка,
    // сравнивается с индексом максимума в соответствующем one-hot векторе y.
    double evaluate(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
        Eigen::MatrixXd predictions = predict(X);
        const int numSamples = predictions.cols();
        int correct = 0;
        for (int i = 0; i < numSamples; i++) {
            // Для вектора-примера выбираем индекс максимального элемента
            Eigen::Index predictedLabel, trueLabel;
            predictions.col(i).maxCoeff(&predictedLabel);
            y.col(i).maxCoeff(&trueLabel);
            if (predictedLabel == trueLabel)
                correct++;
        }
        double accuracy = (static_cast<double>(correct) / numSamples) * 100.0;
        return accuracy;
    }
};
