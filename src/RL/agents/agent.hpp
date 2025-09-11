#pragma once

#include "../basic/basic.hpp"
#include "../../math/math.hpp"

namespace MCL::RL
{
    enum class AgentType
    {
        Normal = 0,
        Policy = 1 << 0,
        Value = 1 << 1,
    };

    AgentType operator|(AgentType, AgentType);

    /**
     * @brief verify the "target" has the attribute "type"
     *
     */
    bool hasType(AgentType target, AgentType type);

    class Agent
    {
    public:
        virtual math::Real update(const std::vector<Transition> &) = 0;

        virtual AgentType type() const;
    };

    namespace Agents
    {
        template <typename DeriveredAgent>
        DeriveredAgent *cast(Agent *);

        template <typename DeriveredAgent>
        const DeriveredAgent *cast(const Agent *);
    }
}