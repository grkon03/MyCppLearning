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
        virtual void setState(StateType) = 0;
    };

    class VectorState : public State<math::Rmatrix>
    {
    public:
        using State::StateType;

    private:
        StateType state;

    public:
        VectorState();
        VectorState(StateType state);
        VectorState(const VectorState &);

        virtual StateType getState() const override;
        virtual void setState(StateType state) override;
    };
}