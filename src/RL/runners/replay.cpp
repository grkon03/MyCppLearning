#include "replay.hpp"

namespace MCL::RL::Runners
{
    ReplayRunner::ReplayRunner(size_t capacityOfReplayBuffer)
        : replayBuffer(capacityOfReplayBuffer), rndgen(std::random_device()()) {}

    std::vector<math::Real> ReplayRunner::play(
        Environment *env, Agent *agent, Searcher *searcher,
        size_t noEpisodes, size_t updatePeriod, size_t batchsize, size_t startsize, size_t epochsToTrain)
    {
        std::vector<math::Real> losses;
        size_t i = 0, epochs, j = 0, noneUpdateCount = 0;
        for (i = 0; i < noEpisodes; ++i)
        {
            ++noneUpdateCount;
            for (auto t : searcher->makeEpisode(env, agent).transitions)
            {
                replayBuffer.push(t);
            }

            if (noneUpdateCount == updatePeriod)
            {
                if (replayBuffer.getSize() > startsize)
                    for (epochs = 0; epochs < epochsToTrain; ++epochs)
                    {
                        for (auto batch : replayBuffer.getBatches(batchsize))
                        {
                            agent->update(batch);
                        }
                    }
                noneUpdateCount = 0;
            }
        }

        if (noneUpdateCount > 0 && replayBuffer.getSize() > startsize)
        {
            for (epochs = 0; epochs < epochsToTrain; ++epochs)
            {
                for (auto batch : replayBuffer.getBatches(batchsize))
                {
                    agent->update(batch);
                }
            }
        }

        return losses;
    }
}