#ifndef MNISTLOADER_H
#define MNISTLOADER_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

class MNISTLoader {
public:
    // Изображения хранятся в матрице размера (numPixels, numExamples)
    Eigen::MatrixXd images;
    // Метки хранятся в матрице размера (10, numExamples) – one-hot представление
    Eigen::MatrixXd labels;

    // Загружает данные из файлов изображений и меток.
    // Если maxExamples == 0, загружаются все доступные примеры.
    bool loadData(const std::string &imageFilename, const std::string &labelFilename, size_t maxExamples = 0) {
        return loadImages(imageFilename, maxExamples) && loadLabels(labelFilename, maxExamples);
    }

private:
    // Вспомогательная функция для чтения 32-битного числа в формате big-endian
    uint32_t readBigEndian(std::ifstream &ifs) {
        uint32_t result = 0;
        unsigned char bytes[4];
        ifs.read(reinterpret_cast<char*>(bytes), 4);
        result = (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
                 (uint32_t(bytes[2]) << 8)  | uint32_t(bytes[3]);
        return result;
    }

    // Загружает изображения из файла.
    // Результат сохраняется в матрице images, где каждый столбец — отдельное изображение.
    bool loadImages(const std::string &filename, size_t maxExamples) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "Не удалось открыть файл изображений: " << filename << std::endl;
            return false;
        }
        uint32_t magic = readBigEndian(ifs);
        if (magic != 2051) { // MNIST magic number для изображений
            std::cerr << "Неверный magic number для файла изображений: " << magic << std::endl;
            return false;
        }
        uint32_t numImages = readBigEndian(ifs);
        uint32_t numRows = readBigEndian(ifs);
        uint32_t numCols = readBigEndian(ifs);
        size_t imageSize = numRows * numCols;
        uint32_t numImagesToLoad = (maxExamples > 0 && maxExamples < numImages)
                                        ? static_cast<uint32_t>(maxExamples)
                                        : numImages;

        images.resize(imageSize, numImagesToLoad);

        for (uint32_t i = 0; i < numImagesToLoad; i++) {
            std::vector<unsigned char> buffer(imageSize);
            ifs.read(reinterpret_cast<char*>(buffer.data()), imageSize);
            if (ifs.gcount() != static_cast<std::streamsize>(imageSize)) {
                std::cerr << "Ошибка чтения данных для изображения " << i << std::endl;
                return false;
            }
            for (size_t j = 0; j < imageSize; j++) {
                images(j, i) = static_cast<double>(buffer[j]) / 255.0;
            }
        }
        return true;
    }

    // Загружает метки из файла.
    // Результат сохраняется в матрице labels, где каждый столбец – one-hot вектор длины 10.
    bool loadLabels(const std::string &filename, size_t maxExamples) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "Не удалось открыть файл меток: " << filename << std::endl;
            return false;
        }
        uint32_t magic = readBigEndian(ifs);
        if (magic != 2049) { // MNIST magic number для меток
            std::cerr << "Неверный magic number для файла меток: " << magic << std::endl;
            return false;
        }
        uint32_t numLabels = readBigEndian(ifs);
        uint32_t numLabelsToLoad = (maxExamples > 0 && maxExamples < numLabels)
                                        ? static_cast<uint32_t>(maxExamples)
                                        : numLabels;


        labels.resize(10, numLabelsToLoad);
        labels.setZero();

        for (uint32_t i = 0; i < numLabelsToLoad; i++) {
            unsigned char label = 0;
            ifs.read(reinterpret_cast<char*>(&label), 1);
            if (ifs.gcount() != 1) {
                std::cerr << "Ошибка чтения метки " << i << std::endl;
                return false;
            }
            labels(static_cast<int>(label), i) = 1.0;
        }
        return true;
    }
};

#endif // MNISTLOADER_H