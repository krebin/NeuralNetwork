//
// Created by krebin on 6/22/21.
//

#include "Network.h"
#include "Linear.h"
#include "Activation.h"


Network::Network()
{
    this->layers.push_back(new Linear(784, 50));
    this->layers.push_back(new Activation("relu"));
    this->layers.push_back(new Linear(50, 50));
    this->layers.push_back(new Activation("relu"));
    this->layers.push_back(new Linear(50, 10));
    this->layers.push_back(new Activation("softmax"));
}

Network::~Network()
{
    for (auto itr = this->layers.begin(); itr < this->layers.end(); ++itr)
        delete *itr;
}

Tensor<float> Network::forward(Tensor<float> X)
{
    for (auto itr = this->layers.begin(); itr < this->layers.end(); ++itr)
        X = (**itr)(X);

    return X;
}

Tensor<float> Network::operator()(Tensor<float> X)
{
    return this->forward(X);
}

void Network::backward(Tensor<float> delta)
{
    for(auto itr = this->layers.rbegin() + 1; itr < this->layers.rend(); itr++)
        delta = (*itr)->backward(delta);
}

void Network::optimize(float lr)
{
    for(auto itr = this->layers.rbegin() + 1; itr < this->layers.rend(); itr++)
        (*itr)->optimize(lr);
}

