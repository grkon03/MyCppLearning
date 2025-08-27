#pragma once

#include <random>
#include <functional>
#include "agent.hpp"
#include "../util/replaybuffer.hpp"
#include "../../NN/neuralnetwork.hpp"

namespace MCL::RL::Agents
{
    class VNNAgent : public VectorAgent
    {
    public:
        using Agent::Action;
        using Agent::State;

        static constexpr math::Real gammaDefault = 0.9;

        using Transition = struct
        {
            State::StateType state;
            math::Real reward;
            State::StateType nextState;
            bool done;
        };

    protected:
        math::Real gamma;

        NN::NeuralNetwork *vfuncNN;
        NN::NeuralNetwork *vfuncNNtarget;
        NN::LearningEngine *engine;

        std::function<VectorAction *(const State *, const VNNAgent *)> actionSelector;

        util::ReplayBuffer<Transition> replaybuffer;

        size_t batchsize = 1;
        size_t synchronizePeriod = 1;

        size_t asyncUpdateCount = 0;

    public:
        // constructors

        VNNAgent();
        VNNAgent(size_t statesize, size_t actionsize);
        VNNAgent(size_t statesize, size_t actionsize, math::Real gamma);
        VNNAgent(size_t statesize, size_t actionsize, math::Real gamma, NN::NeuralNetwork *vfuncNN, NN::LearningEngine *engine);
        VNNAgent(const VNNAgent &);

        // basic methods

        virtual VectorAction *getAction(const State *) const override;
        virtual math::Real update(const State *state, const Action *action, math::Real reward, const State *nextState, bool done) override;
        virtual VNNAgent *copy() const override;

        bool isPrepared() const;

        // setters

        void setGamma(math::Real);
        void setVfuncNN(NN::NeuralNetwork *);
        void setLearningEngine(NN::LearningEngine *);
        void setActionSelector(std::function<VectorAction *(const State *, const VNNAgent *)>);

        void useReplayBuffer(size_t capacity, size_t batchsize);
        void useTargetNetwork(size_t synchronizePeriod);

        void synchronizeTarget();

        // evaluation

        math::Real evaluation(const State *) const;

        // save

        void save(std::string filename);
    };
}