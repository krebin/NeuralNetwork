//
// Created by krebin on 6/24/21.
//

#ifndef NOORALNETWORK_LOSS_H
#define NOORALNETWORK_LOSS_H

#include "Tensor.h"

class Loss
{
public:
    /**
     * Constructor
     *
     * @param loss Loss type for loss calculation
     */
    Loss(std::string loss="log_crossent");

    /**
     * Destructor
     */
    virtual ~Loss();

    /**
     * Forward pass for loss calculation
     *
     * @param T ground truth values
     * @param Y predicted values
     *
     * @return loss value
     */
    float forward(Tensor<float> T, Tensor<float> Y);

    /**
     * Call operator, calls forward
     *
     * @param T ground truth values
     * @param Y predicted values
     *
     * @return loss value
     */
    float operator() (Tensor<float> T, Tensor<float> Y);

protected:

};

#endif //NOORALNETWORK_LOSS_H
