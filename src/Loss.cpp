//
// Created by krebin on 6/24/21.
//

#include "Loss.h"

Loss::~Loss()
{

}

Loss::Loss(std::string loss)
{

}

float Loss::forward(Tensor<float> T, Tensor<float> Y)
{
    return ((T * Y.ln()).mean() * -1);
}

float Loss::operator()(Tensor<float> T, Tensor<float> Y)
{
    return forward(T, Y);
}

