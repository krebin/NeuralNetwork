//
// Created by krebin on 6/15/21.
//

#ifndef NOORALNETWORK_NETWORK_H
#define NOORALNETWORK_NETWORK_H


#include "Tensor.h"
#include "Layer.h"

/**
 * Neural network
 */
class Network
{
public:
    /**
     * Constructor
     */
    Network();

    /**
     * Destructor
     */
    ~Network();

    /**
     * Call operator
     *
     * Calls forward
     *
     * @return Tensor X after going through all layers
     */
    Tensor<float> operator()(Tensor<float> X);

    /**
     * Puts input through all layers
     *
     * @param X input Tensor
     * @return Tensor X after going through all layers
     */
    Tensor<float> forward(Tensor<float> X);

    /**
     * Backward pass, calculates gradients for all layers
     *
     * @param delta Delta from previous layer
     */
    void backward(Tensor<float> delta);

    /**
     * Optimize weights
     *
     * @param lr learning rate for gradient subraction
     */
    void optimize(float lr=0.001);

    // Layers to put X through
    std::vector<Layer*> layers;


};

#endif //NOORALNETWORK_NETWORK_H