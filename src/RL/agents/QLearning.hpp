#pragma once

#include <random>
#include "agent.hpp"
#include "../../NN/nn.hpp"

namespace MCL::RL::Agents
{
    /**
     * @brief QLearningAgent
     *
     * @note actions are interpreted as one-hot vectors
     *
     */
    class QLearningAgent : public VectorAgent
    {
    public:
        using Agent::Action;
        using Agent::State;

        inline static const math::Real defaultGamma = 0.9;
        inline static const math::Real defaultEpsilon = 0.1;

    protected:
        math::Real gamma;
        math::Real epsilon;
        NN::NeuralNetwork *QfuncNN;
        NN::LearningEngine *engine;

        // for randoms

        mutable std::mt19937_64 rnd;
        mutable std::uniform_real_distribution<double> unifdist01;
        mutable std::uniform_int_distribution<size_t> unifdistAction;

    public:
        // constructors

        QLearningAgent();
        QLearningAgent(size_t statesize, size_t actionsize);
        QLearningAgent(size_t statesize, size_t actionsize, math::Real gamma, math::Real epsilon);
        QLearningAgent(size_t statesize, size_t actionsize, math::Real gamma, math::Real epsilon,
                       NN::NeuralNetwork *QfuncNN, NN::LearningEngine *engine);

        // basic methods

        virtual VectorAction *getAction(State *) const override;
        virtual math::Real update(State *state, Action *action, math::Real reward, State *nextState, bool done) override;

        virtual QLearningAgent *copy() const override;

        // setters

        void setGamma(math::Real);
        void setEpsilon(math::Real);
        void setQFuncNN(NN::NeuralNetwork *);
        void setLearningEngine(NN::LearningEngine *);
        virtual void setActionSize(size_t size) override;

        // getters

        NN::NeuralNetwork *copyQfuncNN() const;
    };
}