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
        scr/Layer.cpp
        scr/Activation.cpp
        scr/Loss.cpp
        include/Optimizer.h
        scr/Optimizer.cpp
        include/NeuralNetwork.h
        include/Builder.h
        include/MNISTLoader.h
        include/ConsoleApp.h
        scr/ConsoleApp.cpp
)

add_subdirectory(libs/eigen)

target_link_libraries(kp_NeuralNetwork PRIVATE Eigen3::Eigen)


