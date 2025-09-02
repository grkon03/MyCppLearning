#pragma once

#include "../basic/basic.hpp"
#include "../../math/math.hpp"

namespace MCL::RL
{
    enum class EnvironmentType : size_t
    {
        Normal = 0,
        Discrete = 1 << 0,
    };

    EnvironmentType operator|(EnvironmentType, EnvironmentType);

    /**
     * @brief verify the "target" has the attribute "type"
     *
     */
    bool hasType(EnvironmentType target, EnvironmentType type);

    class Environment
    {
    public:
        struct StepReturn
        {
            math::Rmatrix stateVector;
            math::Real reward;
            math::Rmatrix nextStateVector;
            bool done;
        };
        virtual StepReturn step(Action) = 0;
        virtual std::unique_ptr<Environment> copy() const = 0;
        virtual State state() const = 0;
        virtual State reset() const = 0;
        virtual bool done() const = 0;

        virtual EnvironmentType type() const;
    };

    namespace Environments
    {
        template <typename DeriveredEnvironment>
        DeriveredEnvironment *cast(Environment *env);

        template <typename DeriveredEnvironment>
        const DeriveredEnvironment *cast(const Environment *env);
    }
}