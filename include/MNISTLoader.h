#ifndef MNISTLOADER_H
#define MNISTLOADER_H

#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace NN {

class MNISTLoader {
 public:
  Eigen::MatrixXd images;
  Eigen::MatrixXd labels;

  void loadData(const std::string &imageFilename,
                const std::string &labelFilename, size_t maxExamples = 0);

 private:
  uint32_t readBigEndian(std::ifstream &ifs);
  void loadImages(const std::string &filename, size_t maxExamples);
  void loadLabels(const std::string &filename, size_t maxExamples);
};

}  // namespace NN

#endif  // MNISTLOADER_H
