#include "../include/MNISTLoader.h"
#include <cassert>
#include <iostream>

namespace NN {

uint32_t MNISTLoader::readBigEndian(std::ifstream &ifs) {
    unsigned char bytes[4];
    ifs.read(reinterpret_cast<char*>(bytes), 4);
    assert(ifs.gcount() == 4 && "Failed to read 4 bytes for big-endian integer");
    return (uint32_t(bytes[0]) << 24) |
           (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8)  |
           uint32_t(bytes[3]);
}

void MNISTLoader::loadData(const std::string &imageFilename, const std::string &labelFilename, size_t maxExamples) {
    assert(!imageFilename.empty() && "Image filename is empty");
    assert(!labelFilename.empty() && "Label filename is empty");
    loadImages(imageFilename, maxExamples);
    loadLabels(labelFilename, maxExamples);
}

void MNISTLoader::loadImages(const std::string &filename, size_t maxExamples) {
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open() && "Cannot open image file");

    uint32_t magic = readBigEndian(ifs);
    assert(magic == 2051 && "Invalid magic number for image file");

    uint32_t numImages = readBigEndian(ifs);
    uint32_t numRows   = readBigEndian(ifs);
    uint32_t numCols   = readBigEndian(ifs);
    size_t imageSize   = static_cast<size_t>(numRows) * numCols;
    uint32_t toLoad    = (maxExamples > 0 && maxExamples < numImages)
                             ? static_cast<uint32_t>(maxExamples)
                             : numImages;

    images.resize(imageSize, toLoad);

    for (uint32_t i = 0; i < toLoad; ++i) {
        std::vector<unsigned char> buffer(imageSize);
        ifs.read(reinterpret_cast<char*>(buffer.data()), imageSize);
        assert(ifs.gcount() == static_cast<std::streamsize>(imageSize) && "Failed to read full image data");

        for (size_t j = 0; j < imageSize; ++j) {
            images(j, i) = static_cast<double>(buffer[j]) / 255.0;
        }
    }
}

void MNISTLoader::loadLabels(const std::string &filename, size_t maxExamples) {
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open() && "Cannot open label file");

    uint32_t magic = readBigEndian(ifs);
    assert(magic == 2049 && "Invalid magic number for label file");

    uint32_t numLabels = readBigEndian(ifs);
    uint32_t toLoad    = (maxExamples > 0 && maxExamples < numLabels)
                             ? static_cast<uint32_t>(maxExamples)
                             : numLabels;

    labels.resize(10, toLoad);
    labels.setZero();

    for (uint32_t i = 0; i < toLoad; ++i) {
        unsigned char label;
        ifs.read(reinterpret_cast<char*>(&label), 1);
        assert(ifs.gcount() == 1 && "Failed to read label byte");
        assert(label < 10 && "Label value out of range");

        labels(static_cast<int>(label), i) = 1.0;
    }
}

} // namespace NN