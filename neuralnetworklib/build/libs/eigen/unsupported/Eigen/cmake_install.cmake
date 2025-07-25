# Install script for directory: /Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/AdolcForward"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/AlignedVector3"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/ArpackSupport"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/AutoDiff"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/BVH"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/EulerAngles"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/FFT"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/IterativeSolvers"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/KroneckerProduct"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/MatrixFunctions"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/MPRealSupport"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/NNLS"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/NonLinearOptimization"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/NumericalDiff"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/OpenGLSupport"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/Polynomials"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/SparseExtra"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/SpecialFunctions"
    "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/libs/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/build/libs/eigen/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/tulinovtim/hse/kp-NeuralNetwork/neuralnetworklib/build/libs/eigen/unsupported/Eigen/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
