//
// Created by krebin on 6/24/21.
//

#ifndef NOORALNETWORK_LOSS_H
#define NOORALNETWORK_LOSS_H

#include "Tensor.h"

class Loss
{
public:
    Loss(std::string loss="log_crossent");
    virtual ~Loss();

    float forward(Tensor<float> T, Tensor<float> Y);
    float operator() (Tensor<float> T, Tensor<float> Y);

protected:

};

#endif //NOORALNETWORK_LOSS_H
