#include "value.hpp"

namespace MCL::RL::Agents
{
    AgentType ValueAgent::type() const { return AgentType::Value; }

    template <>
    ValueAgent *cast<ValueAgent>(Agent *a)
    {
        assert(hasType(a->type(), AgentType::Value));
        return static_cast<ValueAgent *>(a);
    }

    template <>
    const ValueAgent *cast<ValueAgent>(const Agent *a)
    {
        assert(hasType(a->type(), AgentType::Value));
        return static_cast<const ValueAgent *>(a);
    }
}