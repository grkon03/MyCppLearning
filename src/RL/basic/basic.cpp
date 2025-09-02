#include "basic.hpp"

namespace MCL::RL
{
    DiscreteAction::operator Action() const
    {
        auto action = Action{math::Rmatrix(size, 1, 0)};
        for (auto hotbit : hotbits)
        {
            action.vectorexp.direct(hotbit) = 1;
        }

        return action;
    }
}