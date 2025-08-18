#include "QLearning.hpp"

namespace MCL::RL::Agents
{
    QLearningAgent::QLearningAgent() : QLearningAgent(0, 0) {}
    QLearningAgent::QLearningAgent(size_t statesize, size_t actionsize)
        : QLearningAgent(statesize, actionsize, defaultGamma, defaultEpsilon) {}
    QLearningAgent::QLearningAgent(size_t statesize, size_t actionsize, math::Real gamma, math::Real epsilon)
        : QLearningAgent(statesize, actionsize, gamma, epsilon, nullptr, nullptr) {}
    QLearningAgent::QLearningAgent(
        size_t statesize, size_t actionsize, math::Real gamma, math::Real epsilon,
        NN::NeuralNetwork *QfuncNN, NN::LearningEngine *engine)
        : VectorAgent(statesize, actionsize), gamma(gamma), epsilon(epsilon),
          QfuncNN(QfuncNN), engine(engine), rnd(std::random_device{}()), unifdist01(0, 1), unifdistAction(0, actionsize - 1) {}

    void QLearningAgent::setGamma(math::Real _gamma) { gamma = _gamma; }
    void QLearningAgent::setEpsilon(math::Real _epsilon) { epsilon = _epsilon; }
    void QLearningAgent::setQFuncNN(NN::NeuralNetwork *qfuncNN) { QfuncNN = qfuncNN; }
    void QLearningAgent::setLearningEngine(NN::LearningEngine *_engine) { engine = _engine; }
    void QLearningAgent::setActionSize(size_t size)
    {
        unifdistAction = std::uniform_int_distribution<size_t>(0, size - 1);
    }

    QLearningAgent *QLearningAgent::copy() const
    {
        return new QLearningAgent(statesize, actionsize, gamma, epsilon, QfuncNN->copy(), engine->copy());
    }

    VectorAction *QLearningAgent::getAction(State *state) const
    {
        math::Rmatrix action = math::Rmatrix(actionsize, 1, 0);
        if (unifdist01(rnd) < epsilon)
        {
            size_t i = std::uniform_int_distribution<int>(0, actionsize - 1)(rnd);
            action.at(i, 0) = 1;
        }
        else
        {
            auto Qs = QfuncNN->predict(state->getState());
            auto [i, j] = Qs.argmax();
            action.at(i, j) = 1;
        }

        return new VectorAction(action);
    }

    math::Real QLearningAgent::update(State *state, Action *action, math::Real reward, State *nextState, bool done)
    {
        math::Real nextQ = 0;
        if (!done)
        {
            auto nextQs = QfuncNN->predict(nextState->getState());
            nextQ = nextQs.max();
        }

        auto target = gamma * nextQ + reward;
        auto Qs = QfuncNN->predict(state->getState());
        QfuncNN->learn(engine, math::Rmatrix(target));

        return QfuncNN->loss();
    }

    NN::NeuralNetwork *QLearningAgent::copyQfuncNN() const
    {
        return QfuncNN->copy();
    }
}