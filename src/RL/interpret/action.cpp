#include "action.hpp"

namespace MCL::RL::interpret::action
{
    math::Real probabilityOfActionWithPolicy(math::Rmatrix policy, DiscreteAction action)
    {
        math::Real prob = 1;
        for (auto bit : action.hotbits)
        {
            prob *= policy.direct(bit);
        }

        return prob;
    }
}