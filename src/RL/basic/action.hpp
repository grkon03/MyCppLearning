#pragma once

#include "../../math/math.hpp"

namespace MCL::RL
{
    template <typename _ActionType>
    class Action
    {
    public:
        using ActionType = _ActionType;

        virtual ActionType getAction() const = 0;
        virtual void setAction(ActionType) = 0;
    };

    class VectorAction : public Action<math::Rmatrix>
    {
    public:
        using Action::ActionType;

    protected:
        ActionType action;

    public:
        VectorAction();
        VectorAction(ActionType);
        VectorAction(const VectorAction &);

        virtual ActionType getAction() const override;
        virtual void setAction(ActionType) override;
    };
}