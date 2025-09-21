#pragma once

#include "value.hpp"
#include "policy.hpp"

#include <tuple>

namespace MCL::RL::Agents
{
    class PVAgent : public Agent
    {
    public:
        virtual math::Real value(const State &) const = 0;
        virtual math::Rmatrix policy(const State &) const = 0;
        virtual std::pair<math::Rmatrix, math::Real> policyvalue(const State &) const;

        virtual AgentType type() const;
    };
}