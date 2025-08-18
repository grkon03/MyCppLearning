#pragma once

#include "../basic/basic.hpp"
#include "../util/replaybuffer.hpp"
#include "QLearning.hpp"

namespace MCL::RL::Agents
{
    class DQNAgent : public QLearningAgent
    {
    public:
        inline static const size_t defaultBatchsize = 64;

        using Agent::Action;
        using Agent::State;

        struct Transition
        {
            State *state;
            Action *action;
            math::Real reward;
            State *nextState;
            bool done;
        };
        using ReplayBuffer = util::ReplayBuffer<Transition>;

    protected:
        size_t batchsize;
        ReplayBuffer replayBuffer;
        NN::NeuralNetwork *QfuncNNtarget;

    public:
        // constructors

        DQNAgent();
        DQNAgent(size_t statesize, size_t actionsize);
        DQNAgent(size_t statesize, size_t actionsize, size_t batchsize, size_t replayBufferCapacity, math::Real gamma, math::Real epsilon);
        DQNAgent(size_t statesize, size_t actionsize, size_t batchsize, size_t replayBufferCapacity, math::Real gamma, math::Real epsilon,
                 NN::NeuralNetwork *QfuncNN, NN::LearningEngine *engine);
        DQNAgent(QLearningAgent qlagent, size_t batchsize, size_t replayBufferCapacity);

        // basic methods

        virtual math::Real update(State *state, Action *action, math::Real reward, State *nextState, bool done) override;

        virtual DQNAgent *copy() const override;

        // setters

        void setBatchSize(size_t);
        void resizeReplayBufferCapacity(size_t);
        void synchronizeQfunc();
    };
}