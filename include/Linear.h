//
// Created by krebin on 6/21/21.
//

#ifndef NOORALNETWORK_LINEAR_H
#define NOORALNETWORK_LINEAR_H


#include "Layer.h"

class Linear: public Layer
{
public:
    Linear(int in_dims, int out_dims, float* weights=NULL);
    virtual ~Linear();

    virtual Tensor<float> forward(Tensor<float> X);
    virtual Tensor<float> backward(Tensor<float> delta);
    virtual Tensor<float> operator() (Tensor<float> X);
    virtual void optimize(float lr=0.001);

    Tensor<float> *_weights;
    Tensor<float> *_bias;

    Tensor<float> *_dx;
    Tensor<float> *_dw;
    Tensor<float> *_db;

protected:

};

#endif //NOORALNETWORK_LINEAR_H