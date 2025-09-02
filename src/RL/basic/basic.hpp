#pragma once

#include <set>
#include "../../math/matrix.hpp"

namespace MCL::RL
{
    struct State
    {
        math::Rmatrix vectorexp;
    };

    struct Action
    {
        math::Rmatrix vectorexp;
    };

    struct DiscreteAction
    {
        std::set<size_t> hotbits;
        size_t size;

        operator Action() const;
    };

    struct DiscreteActionHash
    {
        size_t operator()(const DiscreteAction &s)
        {
            std::size_t h = 0;
            std::hash<size_t> hasher;
            for (const auto &elem : s.hotbits)
            {
                h ^= hasher(elem) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

    struct DiscreteActionEqual
    {
        bool operator()(const std::set<size_t> &a, const std::set<size_t> &b) const
        {
            return a == b;
        }
    };

    struct Transition
    {
        math::Rmatrix stateVector;
        math::Rmatrix actionVector;
        math::Real reward;
        math::Rmatrix nextStateVector;
        bool done;
    };

    struct Episode
    {
        std::vector<Transition> transitions;
    };
}