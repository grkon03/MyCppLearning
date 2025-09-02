#pragma once

#include "agent.hpp"

namespace MCL::RL::Agents
{

    class PolicyAgent : public Agent
    {
    public:
        // return distribution of action
        virtual math::Rmatrix policy(const State &) const = 0;

        virtual AgentType type() const override;
    };
}