//
// Created by krebin on 6/22/21.
//

#ifndef NOORALNETWORK_ACTIVATION_H
#define NOORALNETWORK_ACTIVATION_H

#include "Layer.h"

class Activation: public Layer
{
public:
    Activation(std::string activation="relu");
    virtual ~Activation();

    virtual Tensor<float> forward(Tensor<float> X);
    virtual Tensor<float> backward(Tensor<float> delta);
    virtual Tensor<float> operator() (Tensor<float> X);

    std::string activation;

};

#endif //NOORALNETWORK_ACTIVATION_H