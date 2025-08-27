#pragma once

#include "../basic/basic.hpp"
#include "../agents/agent.hpp"
#include "../environments/environment.hpp"

namespace MCL::RL::Runners
{
    class VectorRunner
    {
    public:
        using State = VectorState;
        using Action = VectorAction;
        using Agent = VectorAgent;
        using Environment = VectorEnvironment;
        using StepReturn = Environment::StepReturn;

    private:
        Agent *agent;
        Environment *environment;

    public:
        // constructors

        VectorRunner();
        VectorRunner(Agent *, Environment *);

        // training

        void trainEpisodes(size_t noEpisodes);
        void trainEpisodesMC(size_t noEpisodes);
        void trainLossThresholds(math::Real thresholds, size_t maxEpisodes);

        void setAgent(Agent *);
        void setEnvironment(Environment *);
    };
}