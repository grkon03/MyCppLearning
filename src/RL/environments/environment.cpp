#include "environment.hpp"

namespace MCL::RL
{
    EnvironmentType operator|(EnvironmentType a, EnvironmentType b)
    {
        return EnvironmentType(size_t(a) | size_t(b));
    }

    bool hasType(EnvironmentType target, EnvironmentType type)
    {
        return (size_t(target) & size_t(type)) == size_t(type);
    }

    EnvironmentType Environment::type() const { return EnvironmentType::Normal; }
}