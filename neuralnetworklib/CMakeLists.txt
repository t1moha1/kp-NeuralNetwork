cmake_minimum_required(VERSION 3.15)
project(neuralnetworklib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) Подключаем Eigen-сабмодуль
add_subdirectory(libs/eigen)

# 2) Объявляем статическую библиотеку neuralnetworklib
add_library(neuralnetworklib STATIC
    src/NeuralNetwork.cpp
    src/MNISTLoader.cpp
    src/Builder.cpp
    src/Layer.cpp
    src/Activation.cpp
    src/Loss.cpp
    src/Optimizer.cpp
)

# 3) Заголовочные директории для вашей библиотеки
target_include_directories(neuralnetworklib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 4) Линкуем Eigen к вашей библиотеке
target_link_libraries(neuralnetworklib PUBLIC Eigen3::Eigen)

# 5) Правила установки (опционально)
install(TARGETS neuralnetworklib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)