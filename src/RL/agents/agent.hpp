#pragma once

#include "../../math/math.hpp"
#include "../basic/basic.hpp"

namespace MCL::RL
{
    template <typename StateType, typename ActionType>
    class Agent
    {
    public:
        using Action = Action<ActionType>;
        using State = State<StateType>;
        virtual Action *getAction(const State *) const = 0;
        virtual math::Real update(const State *state, const Action *action, math::Real reward, const State *nextState, bool done) = 0;
        virtual Agent<StateType, ActionType> *copy() const = 0;
    };

    class VectorAgent : public Agent<VectorState::StateType, VectorAction::ActionType>
    {
    protected:
        size_t statesize;
        size_t actionsize;

    public:
        VectorAgent();
        VectorAgent(size_t statesize, size_t actionsize);

        virtual void setStateSize(size_t);
        virtual void setActionSize(size_t);

        size_t getStateSize() const;
        size_t getActionSize() const;

        virtual VectorAction *getAction(const State *) const = 0;
        virtual math::Real update(const State *state, const Action *action, math::Real reward, const State *nextState, bool done) = 0;
        virtual VectorAgent *copy() const = 0;
    };
}