#pragma once

#include "../basic/basic.hpp"
#include "../../math/math.hpp"

namespace MCL::RL
{
    template <typename StateType, typename ActionType>
    class Environment
    {
    public:
        using State = State<StateType>;
        using Action = Action<ActionType>;

        using StepReturn = struct
        {
            math::Real reward;
            bool done;
        };

        virtual StepReturn step(Action *action) = 0;
        virtual const State *state() const = 0;
        virtual const State *reset() = 0;
    };

    class VectorEnvironment : public Environment<VectorState::StateType, VectorAction::ActionType>
    {
    public:
        using Environment::Action;
        using Environment::State;

        using Environment::StepReturn;

        virtual StepReturn step(Action *action) override = 0;
        virtual const VectorState *state() const override = 0;
        virtual const VectorState *reset() override = 0;
    };
}