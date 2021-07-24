//
// Created by krebin on 6/22/21.
//

#ifndef NOORALNETWORK_ACTIVATION_H
#define NOORALNETWORK_ACTIVATION_H

#include "Layer.h"

/**
 * Nonlinearity/Activation layer
 */
class Activation: public Layer
{
public:
    /**
     * Constructor
     *
     * @param activation Choice of activation
     */
    Activation(std::string activation="relu");

    /**
     * Destructor
     */
    virtual ~Activation();

    /**
     * Puts input through all layers
     *
     * @param X input Tensor
     * @return Tensor X after going through all layers
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
     * Unused
     */
    virtual void optimize(float lr=0.001);

    std::string activation;

};

#endif //NOORALNETWORK_ACTIVATION_H