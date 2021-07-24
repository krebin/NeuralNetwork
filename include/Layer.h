//
// Created by krebin on 6/15/21.
//

#ifndef NOORALNETWORK_LAYER_H
#define NOORALNETWORK_LAYER_H


#include "Tensor.h"

/**
 * Abstract class to be inherited for other layers
 */
class Layer
{
public:
    /**
     * Constructor
     *
     * @param _in Save input for backward pass
     */
    Layer(): _in(NULL){}

    /**
     * Destructor
     */
    virtual ~Layer() {}

    /**
     * Forward pass
     *
     * @param X input tensor
     *
     * @return output after putting X through layer
     */
    virtual Tensor<float> forward(Tensor<float> X) = 0;

    /**
     * Backward pass
     *
     * @param delta from previous layer
     */
    virtual Tensor<float> backward(Tensor<float> delta) = 0;

    /**
     * Call operator, calls forward
     *
     * @param X input tensor
     *
     * @return output after putting X through layer
     */
    virtual Tensor<float> operator() (Tensor<float> X) = 0;

    /**
      * Optimize weights
      *
      * @param lr learning rate for gradient subraction
      */
    virtual void optimize(float lr=0.001) = 0;

protected:
    Tensor<float> *_in;
};


#endif //NOORALNETWORK_LAYER_H