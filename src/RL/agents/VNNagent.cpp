#include "VNNagent.hpp"

namespace MCL::RL::Agents
{
    namespace
    {
        std::mt19937 gen = std::mt19937(std::random_device()());

        VectorAction *__randomOnehotAction(VNNAgent::State *s, const VNNAgent *agent)
        {
            math::Rmatrix actionV(agent->getActionSize(), 1, 0);
            std::uniform_int_distribution<size_t> actionDist(0, agent->getActionSize() - 1);

            // select randomly
            actionV.at(actionDist(gen), 0) = 1;

            return new VectorAction(actionV);
        }
    }

    VNNAgent::VNNAgent() : VNNAgent(0, 0) {}
    VNNAgent::VNNAgent(size_t statesize, size_t actionsize) : VNNAgent(statesize, actionsize, VNNAgent::gammaDefault) {}
    VNNAgent::VNNAgent(size_t statesize, size_t actionsize, math::Real gamma)
        : VNNAgent(statesize, actionsize, gamma, nullptr, nullptr) {}
    VNNAgent::VNNAgent(size_t statesize, size_t actionsize, math::Real gamma, NN::NeuralNetwork *vfuncNN, NN::LearningEngine *engine)
        : VectorAgent(statesize, actionsize), gamma(gamma), vfuncNN(vfuncNN), engine(engine), actionSelector(__randomOnehotAction)
    {
        if (vfuncNN != nullptr)
            assert(vfuncNN->outputSize() != 0);
    }

    bool VNNAgent::isPrepared() const
    {
        if (
            vfuncNN == nullptr ||
            engine == nullptr ||
            statesize == 0 ||
            actionsize == 0)
            return false;

        return (
            vfuncNN->inputSize() == statesize &&
            vfuncNN->outputSize() == 1);
    }

    void VNNAgent::setGamma(math::Real _g)
    {
        gamma = _g;
    };

    void VNNAgent::setVfuncNN(NN::NeuralNetwork *_v)
    {
        assert(_v != nullptr);
        assert(_v->outputSize() != 0);
        vfuncNN = _v;
    };

    void VNNAgent::setLearningEngine(NN::LearningEngine *_e)
    {
        engine = _e;
    };

    void VNNAgent::setActionSelector(std::function<VectorAction *(State *, const VNNAgent *)> _f)
    {
        actionSelector = _f;
    }

    VectorAction *VNNAgent::getAction(VNNAgent::State *state) const
    {
        return actionSelector(state, this);
    }

    math::Real VNNAgent::update(State *state, Action *action, math::Real reward, State *nextState, bool done)
    {
        assert(isPrepared());
        auto nexteval = vfuncNN->predict(nextState->getState());

        auto target = math::Rmatrix(reward) + (done ? math::Rmatrix(1, nexteval.noColumns(), 0) : gamma * nexteval);

        vfuncNN->predict(state->getState());
        vfuncNN->learn(engine, target);

        return vfuncNN->loss();
    }

    VNNAgent *VNNAgent::copy() const
    {
        auto ret = new VNNAgent(statesize, actionsize, gamma, vfuncNN->copy(), engine->copy());
        ret->setActionSelector(actionSelector);
        return ret;
    }
}