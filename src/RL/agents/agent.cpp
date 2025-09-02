#include "agent.hpp"

namespace MCL::RL
{
    AgentType operator|(AgentType a, AgentType b)
    {
        return AgentType(size_t(a) | size_t(b));
    }

    bool hasType(AgentType target, AgentType type)
    {
        return (size_t(target) & size_t(type)) == size_t(type);
    }

    AgentType Agent::type() const { return AgentType::Normal; }
}