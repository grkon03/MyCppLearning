#include "DQN.hpp"

namespace MCL::RL::Agents
{
    DQNAgent::DQNAgent() : QLearningAgent(), batchsize(defaultBatchsize), replayBuffer(), QfuncNNtarget(nullptr) {}
    DQNAgent::DQNAgent(size_t statesize, size_t actionsize)
        : QLearningAgent(statesize, actionsize), batchsize(defaultBatchsize), replayBuffer(), QfuncNNtarget(nullptr) {}
    DQNAgent::DQNAgent(size_t statesize, size_t actionsize, size_t batchsize, size_t replayBufferCapacity,
                       math::Real gamma, math::Real epsilon)
        : DQNAgent(statesize, actionsize, batchsize, replayBufferCapacity, gamma, epsilon, nullptr, nullptr) {}
    DQNAgent::DQNAgent(size_t statesize, size_t actionsize, size_t batchsize, size_t replayBufferCapacity,
                       math::Real gamma, math::Real epsilon, NN::NeuralNetwork *QfuncNN, NN::LearningEngine *engine)
        : QLearningAgent(statesize, actionsize, gamma, epsilon, QfuncNN, engine),
          batchsize(batchsize), replayBuffer(replayBufferCapacity), QfuncNNtarget(new NN::NeuralNetwork(*QfuncNN)) {}
    DQNAgent::DQNAgent(QLearningAgent qlagent, size_t batchsize, size_t replayBufferCapacity)
        : QLearningAgent(*qlagent.copy()), batchsize(batchsize), replayBuffer(replayBufferCapacity),
          QfuncNNtarget(new NN::NeuralNetwork(*QfuncNN)) {}

    void DQNAgent::setBatchSize(size_t _batchsize) { batchsize = _batchsize; }
    void DQNAgent::resizeReplayBufferCapacity(size_t capacity) { replayBuffer.resizeCapacity(capacity); }
    void DQNAgent::synchronizeQfunc() { *QfuncNNtarget = *QfuncNN; }

    math::Real DQNAgent::update(const State *state, const Action *action, math::Real reward, const State *nextState, bool done)
    {
        replayBuffer.push({state, action, reward, nextState, done});

        auto batches = replayBuffer.getBatch(batchsize);
        size_t i;
        math::Real q, nextq, target, lossAve = 0;
        math::Rmatrix Qs;
        for (i = 0; i < batches.size(); ++i)
        {
            Qs = QfuncNN->predict(batches[i].state->getState());
            auto [_i, _j] = batches[i].action->getAction().argmax();
            q = Qs.at(_i, _j);

            if (done)
                nextq = 0;
            else
                nextq = QfuncNNtarget->predict(batches[i].nextState->getState()).max();

            target = reward + gamma * nextq;
            QfuncNN->learn(engine, math::Rmatrix(target));
            lossAve += (QfuncNN->loss() - lossAve) / (i + 1);
        }

        return lossAve;
    }

    DQNAgent *DQNAgent::copy() const
    {
        return new DQNAgent(*QLearningAgent::copy(), batchsize, replayBuffer.getCapacity());
    }
}