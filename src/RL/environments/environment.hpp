#pragma once

#include "../basic/basic.hpp"
#include "../../math/math.hpp"

namespace MCL::RL
{
    template <typename StateType, typename ActionType>
    class Environment
    {
        using State = State<StateType>;
        using Action = Action<ActionType>;

        using StepReturn = struct
        {
            State *nextState;
            math::Real reward;
            bool done;
        };

        virtual StepReturn step(Action *action) = 0;
    };
}