cmake_minimum_required(VERSION 3.15)
project(mainapp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) Добавляем подпроект-библиотеку
add_subdirectory(${CMAKE_SOURCE_DIR}/neuralnetworklib
                 ${CMAKE_BINARY_DIR}/build)

# 2) Создаём исполняемый таргет
add_executable(mainapp main.cpp)

# 3) Прописывем include-путь (чтобы видеть ваши заголовки)
target_include_directories(mainapp PRIVATE
    ${CMAKE_SOURCE_DIR}/neuralnetworklib/include
)

# 4) Линкуем библиотеку
target_link_libraries(mainapp PRIVATE neuralnetworklib)