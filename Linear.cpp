//
// Created by krebin on 6/22/21.
//

#include <random>

#include "Linear.h"
#include "Tensor.h"
#include "utils.h"


Linear::Linear(int in_dims, int out_dims, float* weights)
{
    _in = _out = 0;
    auto vals = get_rand_normal(0, 1, in_dims * out_dims);

    if (weights)
        this->_weights = new Tensor<float>(weights, {1, in_dims, out_dims}, in_dims * out_dims);
    else
        this->_weights = new Tensor<float>(vals.data(), {1, in_dims, out_dims}, in_dims * out_dims);
}

Linear::~Linear()
{
    delete this->_in;
    delete this->_out;
    delete this->_weights;
}

Tensor<float> Linear::forward(Tensor <float> X)
{
    return bmm(X, *(this->_weights));
}

Tensor<float> Linear::backward(Tensor <float> delta)
{
    return *(this->_weights);
}

Tensor<float> Linear::operator()(Tensor<float> X)
{
    this->_in = new Tensor<float>(X);
    this->_out = new Tensor<float>(this->forward(X));

    return *(this->_out);
}