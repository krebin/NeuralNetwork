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
//    std::cout << X;
    return bmm(X, *(this->_weights));
    // return bmm(X, *(this->_weights)) + *(this->_bias);
}

Tensor<float> Linear::backward(Tensor <float> delta)
{
    int batch_size, in_units, out_units;
    batch_size = delta._dims[0];
    out_units = delta._dims[2];
    in_units = this->_in->_dims[2];

//    std::cout << "Linear 1" << std::endl;
//    std::cout << this->_weights->shape();
//    std::cout << this->_in->shape();
//    std::cout << "delta 1 " << delta.shape();



    auto delta_r = delta.reshape({batch_size, out_units});
    auto w_r = this->_weights->reshape({in_units, out_units});
    auto x_r = this->_in->reshape({batch_size, in_units});



//    std::cout << batch_size << " " << in_units << " " << out_units << std::endl;


    this->_dx = new Tensor<float>(bmm(delta_r.unsqueeze(0), w_r.transpose().unsqueeze(0)).reshape({batch_size,
                                                                                                   1,
                                                                                                   in_units}));

//    std::cout << *(_dx);


//    std::cout << x_r.transpose().unsqueeze(1).shape();
//    std::cout << delta_r.unsqueeze(0).shape();
//    std::cout << bmm(x_r.transpose().unsqueeze(1),  delta_r).shape() << bmm(x_r.transpose().unsqueeze(1),  delta_r)._size;
//
//    std::cout << std::endl;

//    std::cout << *(this->_in);

//    std::cout << x_r.transpose().unsqueeze(1);
//    std::cout << "---------" << std::endl;
//    std::cout << delta_r.unsqueeze(0);
//    assert(4==5);

    this->_dw = new Tensor<float>(bmm(x_r.transpose().unsqueeze(1), delta_r.unsqueeze(0)).reshape({1, in_units, out_units}));

//    std::cout << *(_dw);

    this->_db = new Tensor<float>(delta.sum(0));
//    std::cout << *(this->_dw) << std::endl;
//
//    assert(4==5);

//    std::cout << delta.shape();
//    std::cout << this->_db->shape();

//    std::cout << "Linear 2" << std::endl;
//    std::cout << "delta 2 " << this->_dx->shape();
//    std::cout << this->_dw->shape();
//    std::cout << this->_db->shape();

//    std::cout << _dx->shape();
//
//    std::cout << delta;
//    std::cout << "zzzzz" << std::endl;
//    std::cout << *(this->_db);
//    assert(5 == 4);


//    this->_dw = new Tensor<float>(bmm(this->_in->transpose(), delta));
//    this->_db = new Tensor<float>(delta.sum(0));
//
//    std::cout << "Linear" << std::endl;
//    std::cout << delta.shape();
//    std::cout << this->_in->shape();
//    std::cout << this->_in->squeeze().transpose().shape();
//    std::cout << this->_in->transpose().shape();
//    std::cout << this->_dw->shape();
//    std::cout << this->_db->shape();

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

//    std::cout << *(this->_db);
//    std::cout << *(this->_weights);
}
