cmake_minimum_required(VERSION 3.30)
project(kp_NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

add_executable(kp_NeuralNetwork scr/main.cpp
        include/Activation.h
        include/Loss.h
        include/Layer.h
        scr/NeuralNetwork.cpp
        scr/MNISTLoader.cpp
        scr/Builder.cpp
        scr/Layer.cpp)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

target_link_libraries(kp_NeuralNetwork PRIVATE Eigen3::Eigen)


