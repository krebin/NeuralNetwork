//
// Created by krebin on 6/22/21.
//

#include "Activation.h"

Activation::Activation(std::string activation): activation(activation){}

Activation::~Activation()
{
    delete this->_in;
    delete this->_out;
}

Tensor<float> Activation::forward(Tensor<float> X)
{
    float temp[X._size];

    // max(x, 0)
    if (this->activation == "relu")
        for (int i = 0; i < X._size; i++)
            temp[i] = X._values[i] < 0 ? 0 : X._values[i];

    return Tensor<float>(temp, X._dims, X._size);
}

