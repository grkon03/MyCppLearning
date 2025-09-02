#pragma once

#include "agent.hpp"

namespace MCL::RL::Agents
{

    class ValueAgent : public Agent
    {
    public:
        // return distribution of action
        virtual math::Real value(const State &) const = 0;

        virtual AgentType type() const override;
    };
}