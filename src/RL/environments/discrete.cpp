#include "discrete.hpp"

namespace MCL::RL::Environments
{
    EnvironmentType DiscreteActionEnvironment::type() const
    {
        return EnvironmentType::Discrete;
    }

    template <>
    DiscreteActionEnvironment *cast(Environment *e)
    {
        assert(hasType(e->type(), EnvironmentType::Discrete));
        return static_cast<DiscreteActionEnvironment *>(e);
    }

    template <>
    const DiscreteActionEnvironment *cast(const Environment *e)
    {
        assert(hasType(e->type(), EnvironmentType::Discrete));
        return static_cast<const DiscreteActionEnvironment *>(e);
    }

}