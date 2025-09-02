#include "policy.hpp"

namespace MCL::RL::Agents
{
    AgentType PolicyAgent::type() const { return AgentType::Policy; }

    template <>
    PolicyAgent *cast<PolicyAgent>(Agent *agent)
    {
        assert(hasType(agent->type(), AgentType::Policy));
        return static_cast<PolicyAgent *>(agent);
    }

    template <>
    const PolicyAgent *cast<PolicyAgent>(const Agent *agent)
    {
        assert(hasType(agent->type(), AgentType::Policy));
        return static_cast<const PolicyAgent *>(agent);
    }
}