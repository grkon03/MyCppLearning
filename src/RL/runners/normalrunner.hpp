#pragma once

#include "../basic/basic.hpp"
#include "../agents/agent.hpp"
#include "../environments/environment.hpp"

namespace MCL::RL::Runners
{
    template <typename StateType, typename ActionType>
    class NormalRunner
    {
    public:
        using State = State<StateType>;
        using Action = Action<ActionType>;
        using Agent = Agent<StateType, ActionType>;
        using Environment = Environment<StateType, ActionType>;
        using StepReturn = Environment::StepReturn;

    private:
        Agent *agent;
        Environment *environment;

    public:
        // constructors

        NormalRunner();
        NormalRunner(Agent *, Environment *);

        // training

        void trainEpisodes(size_t noEpisodes);
        /**
         * @brief
         *
         * @param thresholds
         * @param maxEpisodes if maxEpisodes = 0, then the loop cannot be breaked until the loss belows thresholds.
         */
        void trainLossThresholds(math::Real thresholds, size_t maxEpisodes);
    };

    template <typename StateType, typename ActionType>
    NormalRunner<StateType, ActionType>::NormalRunner() : NormalRunner(nullptr, nullptr) {}

    template <typename StateType, typename ActionType>
    NormalRunner<StateType, ActionType>::NormalRunner(Agent *agent, Environment *environment)
        : agent(agent), environment(environment) {}

    template <typename StateType, typename ActionType>
    void NormalRunner<StateType, ActionType>::trainEpisodes(size_t noEpisodes)
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
                agent->update(s, a, stepret.reward, stepret.nextState, stepret.done);
                s = stepret.nextState;
                done = stepret.done;
            }
        }
    }

    template <typename StateType, typename ActionType>
    void NormalRunner<StateType, ActionType>::trainLossThresholds(math::Real thresholds, size_t maxEpisodes)
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
                lossave += (lossave - agent->update(s, a, stepret.reward, stepret.nextState, stepret.done)) / (count + 1);
                s = stepret.nextState;
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