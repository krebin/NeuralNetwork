cmake_minimum_required(VERSION 3.19)
project(NooralNetwork CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(NooralNetwork main.cu Layer.h Network.h Tensor.h Tensor.tpp Linear.h Linear.cpp utils.h utils.cpp Activation.h Activation.cpp Network.cpp)

set_target_properties(
        NooralNetwork
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)