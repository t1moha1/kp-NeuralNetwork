#include "../include/Optimizer.h"
#include <cmath>

namespace NN {

    AdamOptimizer::AdamOptimizer(int num_layers,
                                 double learning_rate,
                                 double beta1,
                                 double beta2,
                                 double epsilon)
        : lr(learning_rate)
        , b1(beta1)
        , b2(beta2)
        , eps(epsilon)
        , t(0)
    {
        m_w.resize(num_layers);
        v_w.resize(num_layers);
        m_b.resize(num_layers);
        v_b.resize(num_layers);
    }

    void AdamOptimizer::update(int layer_index,
                               Eigen::MatrixXd& W,
                               Eigen::VectorXd& b,
                               const Eigen::MatrixXd& dW,
                               const Eigen::VectorXd& db)
    {
        ++t;
        if (m_w[layer_index].size() == 0) {
            m_w[layer_index] = Eigen::MatrixXd::Zero(dW.rows(), dW.cols());
            v_w[layer_index] = Eigen::MatrixXd::Zero(dW.rows(), dW.cols());
            m_b[layer_index] = Eigen::VectorXd::Zero(db.size());
            v_b[layer_index] = Eigen::VectorXd::Zero(db.size());
        }

        m_w[layer_index] = b1 * m_w[layer_index] + (1 - b1) * dW;
        v_w[layer_index] = b2 * v_w[layer_index] + (1 - b2) * dW.array().square().matrix();
        m_b[layer_index] = b1 * m_b[layer_index] + (1 - b1) * db;
        v_b[layer_index] = b2 * v_b[layer_index] + (1 - b2) * db.array().square().matrix();

        auto m_w_hat = m_w[layer_index] / (1 - std::pow(b1, t));
        auto v_w_hat = v_w[layer_index] / (1 - std::pow(b2, t));
        auto m_b_hat = m_b[layer_index] / (1 - std::pow(b1, t));
        auto v_b_hat = v_b[layer_index] / (1 - std::pow(b2, t));

        // Элемент‑wise вычитание через .array()
        W.array() -= lr * m_w_hat.array() / (v_w_hat.array().sqrt() + eps);
        b.array() -= lr * m_b_hat.array() / (v_b_hat.array().sqrt() + eps);
    }

}