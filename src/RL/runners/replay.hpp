#pragma once

#include "../../math/math.hpp"
#include "../basic/basic.hpp"
#include "../util/replaybuffer.hpp"
#include "../searchers/searcher.hpp"

namespace MCL::RL::Runners
{
    class ReplayRunner
    {
    public:
    private:
        util::ReplayBuffer<Transition> replayBuffer;

        mutable std::mt19937 rndgen;

    public:
        ReplayRunner(size_t capacityOfReplayBuffer);

        std::vector<math::Real> play(Environment *, Agent *, Searcher *,
                                     size_t noEpisodes, size_t updatePeriod, size_t batchsize, size_t startsize, size_t epochsToTrain);
    };
}