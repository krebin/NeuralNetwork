//
// Created by krebin on 6/22/21.
//

#include <random>
#include "utils.h"

std::vector<float> get_rand_normal(int mean, int std, int samples)
{
    std::normal_distribution<> d(0, 1);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::vector<float> vals;

    for(int n = 0; n < samples; ++n)
        vals.push_back(d(gen));

    return vals;
}
