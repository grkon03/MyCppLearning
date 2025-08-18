#pragma once

#include "../layers/layer.hpp"

namespace MCL::NN
{
    class LearningEngine
    {
    public:
        virtual void run(std::vector<Layer *>) = 0;
        virtual LearningEngine *copy() const = 0;
    };
}