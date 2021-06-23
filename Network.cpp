//
// Created by krebin on 6/22/21.
//

#include "Network.h"
#include "Activation.h"
#include "Linear.h"

Network::Network()
{
    float A[18], B[12], C[32], D[24];
    range(0, 18, A);
    range(0, 12, B);
    range(0, 32, C);
    range(0, 24, D);

    this->layers.push_back(new Linear(3, 4, B));
    this->layers.push_back(new Linear(4, 8, C));
    this->layers.push_back(new Activation());
    this->layers.push_back(new Linear(8, 3, D));
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
