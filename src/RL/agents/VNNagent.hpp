#pragma once

#include <random>
#include <functional>
#include "agent.hpp"
#include "../../NN/neuralnetwork.hpp"

namespace MCL::RL::Agents
{
    class VNNAgent : public VectorAgent
    {
    public:
        using Agent::Action;
        using Agent::State;

        static constexpr math::Real gammaDefault = 0.9;

    private:
        math::Real gamma;

        NN::NeuralNetwork *vfuncNN;
        NN::LearningEngine *engine;

        std::function<VectorAction *(State *, const VNNAgent *)> actionSelector;

    public:
        // constructors

        VNNAgent();
        VNNAgent(size_t statesize, size_t actionsize);
        VNNAgent(size_t statesize, size_t actionsize, math::Real gamma);
        VNNAgent(size_t statesize, size_t actionsize, math::Real gamma, NN::NeuralNetwork *vfuncNN, NN::LearningEngine *engine);

        // basic methods

        virtual VectorAction *getAction(State *) const;
        virtual math::Real update(State *state, Action *action, math::Real reward, State *nextState, bool done);
        virtual VNNAgent *copy() const;

        bool isPrepared() const;

        // setters

        void setGamma(math::Real);
        void setVfuncNN(NN::NeuralNetwork *);
        void setLearningEngine(NN::LearningEngine *);
        void setActionSelector(std::function<VectorAction *(State *, const VNNAgent *)>);
    };
}