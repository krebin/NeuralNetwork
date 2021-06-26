//
// Created by krebin on 6/15/21.
//

#ifndef NOORALNETWORK_NETWORK_H
#define NOORALNETWORK_NETWORK_H


#include "Tensor.h"
#include "Layer.h"

class Network
{
public:
    Network();
    Tensor<float> operator()(Tensor<float> X);
    Tensor<float> forward(Tensor<float> X);
    void backward(Tensor<float> delta);
    void optimize(float lr=0.001);

    std::vector<Layer*> layers;


};

#endif //NOORALNETWORK_NETWORK_H