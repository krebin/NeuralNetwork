//
// Created by krebin on 6/15/21.
//

#ifndef NOORALNETWORK_LAYER_H
#define NOORALNETWORK_LAYER_H

#endif //NOORALNETWORK_LAYER_H

#include "Tensor.h"

class Layer
{
public:
    Layer(): _in(NULL), _out(NULL){}
    virtual Tensor<float> forward(Tensor<float> X) = 0;
    virtual Tensor<float> backward(Tensor<float> delta) = 0;
    virtual Tensor<float> operator() (Tensor<float> X) = 0;

protected:
    Tensor<float> *_in;
    Tensor<float> *_out;
};

