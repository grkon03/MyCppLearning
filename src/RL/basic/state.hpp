#pragma once

#include "../../math/math.hpp"

namespace MCL::RL
{
    template <typename _StateType>
    class State
    {
    public:
        using StateType = _StateType;

        virtual StateType getState() const = 0;
    };

    class VectorState : public State<math::Rmatrix>
    {
    public:
        using State::StateType;

    public:
        VectorState();

        virtual StateType getState() const override;
    };
}