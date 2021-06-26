//
// Created by krebin on 6/22/21.
//

#ifndef NOORALNETWORK_UTILS_H
#define NOORALNETWORK_UTILS_H

#endif //NOORALNETWORK_UTILS_H

#include <iostream>
std::vector<float> get_rand_normal(int mean, int std, int samples);

float max(float *arr, int size);

unsigned char** read_mnist_images(std::string full_path, int number_of_images, int& image_size);
unsigned char* read_mnist_labels(std::string full_path, int number_of_labels);