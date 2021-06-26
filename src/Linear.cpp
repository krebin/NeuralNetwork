//
// Created by krebin on 6/22/21.
//

#include <random>

#include "Linear.h"
#include "Tensor.h"
#include "utils.h"
#include <assert.h>


Linear::Linear(int in_dims, int out_dims, float* weights)
{
    // Randomly initialize weights in (0, 1) uniform distribution
    auto vals = get_rand_normal(0, 1, in_dims * out_dims);

    if (weights)
        this->_weights = new Tensor<float>(weights, {1, in_dims, out_dims}, in_dims * out_dims);
    else
        this->_weights = new Tensor<float>(vals.data(), {1, in_dims, out_dims}, in_dims * out_dims);

    float temp[out_dims] = {0};
    this->_bias = new Tensor<float>(temp, {1, 1, out_dims}, out_dims);
}

Linear::~Linear()
{
    delete this->_in;
    delete this->_weights;
    delete this->_bias;
    delete this->_dx;
    delete this->_dw;
    delete this->_db;
}

Tensor<float> Linear::forward(Tensor <float> X)
{
    _in = new Tensor<float>(X);
    return bmm(X, *(this->_weights)) + *(this->_bias);
}

Tensor<float> Linear::backward(Tensor <float> delta)
{
    int batch_size, in_units, out_units;
    batch_size = delta._dims[0];
    out_units = delta._dims[2];
    in_units = this->_in->_dims[2];

    auto delta_r = delta.reshape({batch_size, out_units});
    auto w_r = this->_weights->reshape({in_units, out_units});
    auto x_r = this->_in->reshape({batch_size, in_units});


    this->_dx = new Tensor<float>(bmm(delta_r.unsqueeze(0), w_r.transpose().unsqueeze(0)).reshape({batch_size,
                                                                                                   1,
                                                                                                   in_units}));

    this->_dw = new Tensor<float>(bmm(x_r.transpose().unsqueeze(1),
                                      delta_r.unsqueeze(0)).reshape({1, in_units, out_units}));

    this->_db = new Tensor<float>(delta.sum(0));


    return *(this->_dx);
}

Tensor<float> Linear::operator()(Tensor<float> X)
{
    this->_in = new Tensor<float>(X);
    this->_out = new Tensor<float>(this->forward(X));

    return *(this->_out);
}

void Linear::optimize(float lr)
{
    this->_weights = new Tensor<float>(*(this->_weights) - (*(this->_dw)).clip(-10, 10) * lr);
    this->_bias = new Tensor<float>(*(this->_bias) - (*(this->_db)).clip(-10, 10) * lr);
}
