#pragma once

#include "../../math/math.hpp"
#include "../environments/discrete.hpp"

namespace MCL::RL::interpret::action
{
    /**
     * @brief calculate the probability (or the visitablity weight) of the action with the policy vector
     *
     * @note
     * if actions are masked in your implementation, then the sum of probabilities of all masked actions can be less than 1
     *
     * @note
     * The implementation of this function is to calculate the products of values of policy at all hotbits.
     * If the size of action.hotbits can be changed in your environment,
     * then the probability of a small size action will be much greater than it of a big size one.
     *
     * @param policy policy
     * @param action discrete action
     * @return math::Real probability
     */
    math::Real probabilityOfActionWithPolicy(math::Rmatrix policy, DiscreteAction action);
}