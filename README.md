# kp-NeuralNetwork

A simple neural network implementation in C++ for training and testing on the MNIST dataset. Built with Eigen for linear algebra and features a console application interface.

## Features

- Fully connected feedforward neural network
- Activation functions: Sigmoid, Softmax, ReLU
- Loss functions: Mean Squared Error (MSE), Cross-Entropy
- Optimizer: Adam
- MNIST data loader (IDX format)
- Console application for interactive network creation, training, testing, and saving/loading
- Example training script in `main.cpp`

## Prerequisites

- C++20 compatible compiler (GCC, Clang, MSVC)
- CMake ≥ 3.30
- Eigen library (included as a submodule)

## Getting Started

Clone the repository and initialize submodules:

```bash
git clone <repository-url>
git submodule update --init --recursive
```

Create a build directory and compile:

```bash
mkdir build && cd build
cmake ..
make
```

This will generate the `kp_NeuralNetwork` executable.

