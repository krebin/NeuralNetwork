//
// Created by krebin on 6/22/21.
//

#include "Activation.h"
#include "utils.h"
#include <math.h>
#include <assert.h>

Activation::Activation(std::string activation): activation(activation){}

Activation::~Activation()
{
    delete this->_in;
}

Tensor<float> Activation::forward(Tensor<float> X)
{
    float temp[X._size];

    // max(x, 0)
    if (this->activation == "relu")
        for (int i = 0; i < X._size; i++)
            temp[i] = X._values[i] < 0 ? 0 : X._values[i];

    else if (this->activation == "softmax")
    {
        int batch_size, num_classes;
        batch_size = X._dims[0];
        num_classes = X._dims[2];

        memcpy(temp, X._values, batch_size * num_classes * sizeof(float));

        // Apply softmax row wise
        for (int i = 0; i < batch_size; i++)
        {
            int offset;
            float m, sum;

            offset = num_classes * i;

            // get max of array
            m = max(X._values + offset, num_classes);
            sum = 0;

            // Subtract row wise max to stop overflow
            for (int j = 0; j < num_classes; j++)
            {
                (temp + offset)[j] = exp((temp + offset)[j] - m);
                sum += (temp + offset)[j];
            }

            // Divide by sum of transformed values to create a probability
            for (int j = 0; j < num_classes; j++)
                (temp + offset)[j] /= sum;
        }
    }

    return Tensor<float>(temp, X._dims, X._size);
}

Tensor<float> Activation::backward(Tensor<float> delta)
{
    float temp[_in->_size];
    memcpy(temp, _in->_values, _in->_size * sizeof(float));

    for(int i = 0; i < _in->_size; i++)
        temp[i] = temp[i] < 0 ? 0 : 1;

    return Tensor<float>(temp, _in->_dims, _in->_size) * delta;
}

Tensor<float> Activation::operator()(Tensor<float> X)
{
    delete this->_in;
    this->_in = new Tensor<float>(X);
    return this->forward(X);
}

void Activation::optimize(float lr) {}
