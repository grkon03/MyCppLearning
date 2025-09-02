#pragma once

#include "environment.hpp"

namespace MCL::RL::Environments
{

    class DiscreteActionEnvironment : public Environment
    {
    public:
        virtual std::vector<DiscreteAction> getPossibleActions() const = 0;

        virtual EnvironmentType type() const;
    };
}