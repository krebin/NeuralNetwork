//
// Created by krebin on 6/21/21.
//

#ifndef NOORALNETWORK_LINEAR_H
#define NOORALNETWORK_LINEAR_H


#include "Layer.h"

class Linear: public Layer
{
public:
    /**
     * Constructor
     *
     * @param in_dims Number of in channels
     * @param out_dims Number of out channels
     * @param weights Pre computed weights if provided
     */
    Linear(int in_dims, int out_dims, float* weights=NULL);

    /**
     * Destructor
     */
    virtual ~Linear();

    /**
     * Forward pass for linear layer
     *
     * @param X input tensor
     *
     * @return output after putting X through layer
     */
    virtual Tensor<float> forward(Tensor<float> X);

    /**
     * Backward pass for linear layer
     *
     * @param delta from previous layer
     */
    virtual Tensor<float> backward(Tensor<float> delta);

    /**
     * Call operator, calls forward
     *
     * @param X input tensor
     *
     * @return output after putting X through layer
     */
    virtual Tensor<float> operator() (Tensor<float> X);

    /**
     * Optimize weights
     *
     * @param lr learning rate for gradient subraction
     */
    virtual void optimize(float lr=0.001);

    Tensor<float> *_weights;
    Tensor<float> *_bias;

    Tensor<float> *_dx;
    Tensor<float> *_dw;
    Tensor<float> *_db;

protected:

};

#endif //NOORALNETWORK_LINEAR_H