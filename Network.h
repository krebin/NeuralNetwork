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
    Tensor<float> backward();

    std::vector<Layer*> layers;


};

#endif //NOORALNETWORK_NETWORK_H