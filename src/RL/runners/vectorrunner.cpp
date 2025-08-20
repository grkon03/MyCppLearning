#include "vectorrunner.hpp"

namespace MCL::RL::Runners
{
    VectorRunner::VectorRunner() : VectorRunner(nullptr, nullptr) {}
    VectorRunner::VectorRunner(Agent *agent, Environment *environment) : agent(agent), environment(environment) {}

    void VectorRunner::trainEpisodes(size_t noEpisodes)
    {
        size_t i;
        StepReturn stepret;
        Action *a;
        State *s, *nexts;
        bool done;

        for (i = 0; i < noEpisodes; ++i)
        {
            done = false;
            s = environment->reset();

            while (done)
            {
                a = agent->getAction(s);
                stepret = environment->step(a);
                nexts = environment->state();
                agent->update(s, a, stepret.reward, s, stepret.done);
                s = nexts;
                done = stepret.done;
            }
        }
    }

    void VectorRunner::trainLossThresholds(math::Real thresholds, size_t maxEpisodes)
    {
        math::Real lossave;
        StepReturn stepret;
        Action *a;
        State *s, *nexts;
        size_t count, maxCount = std::numeric_limits<size_t>::max();
        bool done;
        bool infloop = (maxEpisodes == 0);

        size_t i = 0;
        while (i < maxEpisodes || infloop)
        {
            done = false;
            s = environment->reset();
            lossave = 0;
            count = 0;

            while (done)
            {
                a = agent->getAction(s);
                stepret = environment->step(a);
                nexts = environment->state();
                lossave += (lossave - agent->update(s, a, stepret.reward, nexts, stepret.done)) / (count + 1);
                s = nexts;
                done = stepret.done;

                ++count;
                if (count == maxCount)
                    break;
            }

            if (lossave < thresholds)
                break;
        }
    }
}