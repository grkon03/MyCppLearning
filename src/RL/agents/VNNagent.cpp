#include "VNNagent.hpp"

namespace MCL::RL::Agents
{
    namespace
    {
        std::mt19937 gen = std::mt19937(std::random_device()());

        VectorAction *__randomOnehotAction(const VNNAgent::State *s, const VNNAgent *agent)
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
        : VNNAgent(statesize, actionsize, gamma, new NN::NeuralNetwork(), nullptr) {}
    VNNAgent::VNNAgent(size_t statesize, size_t actionsize, math::Real gamma, NN::NeuralNetwork *vfuncNN, NN::LearningEngine *engine)
        : VectorAgent(statesize, actionsize), gamma(gamma), vfuncNN(vfuncNN), vfuncNNtarget(vfuncNN),
          engine(engine), actionSelector(__randomOnehotAction), replaybuffer(1) {}
    VNNAgent::VNNAgent(const VNNAgent &a)
        : VectorAgent(a.statesize, a.actionsize), gamma(a.gamma), vfuncNN(new NN::NeuralNetwork(*a.vfuncNN)),
          vfuncNNtarget(new NN::NeuralNetwork(*a.vfuncNNtarget)),
          engine(a.engine->copy().release()), actionSelector(a.actionSelector), replaybuffer(a.replaybuffer),
          batchsize(a.batchsize), synchronizePeriod(a.synchronizePeriod), asyncUpdateCount(a.asyncUpdateCount) {}

    bool VNNAgent::isPrepared() const
    {
        if (
            vfuncNN == nullptr ||
            vfuncNNtarget == nullptr ||
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
        if (vfuncNNtarget != vfuncNN)
            *vfuncNNtarget = *vfuncNN;
    };

    void VNNAgent::setLearningEngine(NN::LearningEngine *_e)
    {
        engine = _e;
    };

    void VNNAgent::setActionSelector(std::function<VectorAction *(const State *, const VNNAgent *)> _f)
    {
        actionSelector = _f;
    }

    void VNNAgent::useReplayBuffer(size_t capacity, size_t _batchsize)
    {
        replaybuffer.resizeCapacity(capacity);
        batchsize = _batchsize;
    }

    void VNNAgent::useTargetNetwork(size_t _synchronizePriod)
    {
        synchronizePeriod = _synchronizePriod;
        if (vfuncNNtarget == vfuncNN || vfuncNNtarget == nullptr)
            vfuncNNtarget = new NN::NeuralNetwork(*vfuncNN);
        else if (asyncUpdateCount >= synchronizePeriod)
            synchronizeTarget();
    }

    void VNNAgent::synchronizeTarget()
    {
        if (vfuncNNtarget != vfuncNN)
            *vfuncNNtarget = *vfuncNN;
    }

    VectorAction *VNNAgent::getAction(const State *state) const
    {
        return actionSelector(state, this);
    }

    math::Real VNNAgent::update(const State *state, const Action *action, math::Real reward, const State *nextState, bool done)
    {
        assert(isPrepared());
        replaybuffer.push({state->getState(), reward, nextState->getState(), done});

        auto batch = replaybuffer.getBatch(batchsize);

        for (auto sample : batch)
        {
            auto nexteval = vfuncNNtarget->predict(sample.nextState);
            auto target = math::Rmatrix(sample.reward) + (sample.done ? math::Rmatrix(1, 1, 0) : gamma * nexteval);

            vfuncNN->predict(sample.state);
            vfuncNN->learn(engine, target);

            if (sample.done)
            {
                vfuncNN->predict(sample.nextState);
                vfuncNN->learn(engine, math::Rmatrix(sample.reward));
            }
        }

        ++asyncUpdateCount;
        if (asyncUpdateCount >= synchronizePeriod)
        {
            *vfuncNNtarget = *vfuncNN;
            asyncUpdateCount = 0;
        }

        return vfuncNN->loss();
    }

    VNNAgent *VNNAgent::copy() const
    {
        auto ret = new VNNAgent(statesize, actionsize, gamma, new NN::NeuralNetwork(*vfuncNN), engine->copy().release());
        ret->setActionSelector(actionSelector);
        *(ret->vfuncNNtarget) = *vfuncNNtarget;
        ret->batchsize = batchsize;
        ret->synchronizePeriod = synchronizePeriod;
        ret->asyncUpdateCount = asyncUpdateCount;
        ret->replaybuffer.resizeCapacity(replaybuffer.getCapacity());
        return ret;
    }

    math::Real VNNAgent::evaluation(const State *s) const
    {
        return vfuncNN->predict(s->getState()).direct(0);
    }

    void VNNAgent::save(std::string filename)
    {
        vfuncNN->saveParameters(filename);
    }
}