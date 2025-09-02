#pragma once

#include "value.hpp"
#include "policy.hpp"

#include <tuple>

namespace MCL::RL::Agents
{
    class PVAgent : public PolicyAgent, public ValueAgent
    {
        virtual std::pair<math::Rmatrix, math::Real> policyvalue(const State &) const;

        virtual AgentType type() const;
    };
}