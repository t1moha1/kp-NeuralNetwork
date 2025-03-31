//
// Created by Тимофей Тулинов on 30.03.2025.
//

#ifndef LOSS_H
#define LOSS_H

#include <functional>
#include <Eigen/Dense>
struct Loss {
    std::function<double(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> loss;
    // Функция градиента: принимает матрицы предсказаний и истинных значений, возвращает матрицу градиента
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> derivative;
};


inline double mseLossFunction(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    double n = static_cast<double>(predictions.size());
    return (predictions - targets).array().square().sum() / n;
}

inline Eigen::MatrixXd mseLossDerivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    double n = static_cast<double>(predictions.size());
    return  (predictions - targets) / n;
}

// Кросс-энтропия
// Предполагается, что predictions содержит вероятностное распределение по классам для каждого примера
// (каждый столбец представляет один пример, сумма элементов столбца равна 1),
// а targets представляют one-hot векторы.
// Для числовой устойчивости используется маленькая константа epsilon.
inline double crossEntropyLossFunction(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    const double epsilon = 1e-12;
    // Ограничиваем значения, чтобы избежать log(0)
    Eigen::MatrixXd clipped = predictions.array().max(epsilon).min(1 - epsilon);
    // Вычисляем кросс-энтропию для всех примеров:
    // loss = - (1 / numExamples) * sum(targets * log(clipped))
    int numExamples = predictions.cols();
    double lossSum = -(targets.array() * clipped.array().log()).sum();
    return lossSum / numExamples;
}

// Градиент кросс-энтропии для случая softmax + cross-entropy:
// При такой комбинации градиент упрощается до (predictions - targets)/numExamples.
inline Eigen::MatrixXd crossEntropyLossDerivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    int numExamples = predictions.cols();
    return (predictions - targets) / numExamples;
}

#endif //LOSS_H
