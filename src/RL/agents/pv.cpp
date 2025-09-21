#include "pv.hpp"

namespace MCL::RL::Agents
{
    std::pair<math::Rmatrix, math::Real> PVAgent::policyvalue(const State &state) const
    {
        return {policy(state), value(state)};
    }

    AgentType PVAgent::type() const { return AgentType::Policy | AgentType::Value; }

    template <>
    PVAgent *cast<PVAgent>(Agent *a)
    {
        assert(hasType(a->type(), AgentType::Policy | AgentType::Value));
        return static_cast<PVAgent *>(a);
    }

    template <>
    const PVAgent *cast<PVAgent>(const Agent *a)
    {
        assert(hasType(a->type(), AgentType::Policy | AgentType::Value));
        return static_cast<const PVAgent *>(a);
    }
}