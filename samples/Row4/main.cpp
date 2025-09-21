/**
 * 3-dimensional 4 in a row
 *
 */

#include "row4.hpp"
#include <mcl.hpp>

using namespace MCLSamples::Row4;

math::Real policyInterpreter(math::Rmatrix policy, RL::DiscreteAction action)
{
    return policy.direct(*action.hotbits.begin());
}
math::Real tau(const RL::Searchers::AlphaZeroMCTS::DiscreteEnv *_env)
{
    auto env = static_cast<const Row4Env *>(_env);

    return (double)1 / (env->getBallcount() + 1);
}

int main()
{
    Row4Env env;
    Row4Agent agent;
    RL::Searchers::AlphaZeroMCTS mcts;
    mcts.setting = RL::Searchers::AlphaZeroMCTS::Setting{
        .constantPUCT = 1,
        .noSimulations = 100,
        .gamma = -1,
        .dirichletAlpha = 0.4,
        .dirichletEpsilon = 0.2,
        .tau = tau,
        .tauCutThreshold = 0.1,
        .policyInterpreter = policyInterpreter,
    };

    RL::Runners::ReplayRunner runner(1000);

    std::cout << "Train 1000 episodes" << std::endl;
    runner.play(&env, &agent, &mcts, 1000, 100, 100, 100, 5);

    env.reset();

    // play a game instance

    std::cout << "Play a game instance" << std::endl;
    while (!env.done())
    {
        auto bestmove = agent.getBestAction(&env, &mcts);
        auto bestmoveXY = Row4Env::actionpos(bestmove);
        env.step(bestmoveXY.first, bestmoveXY.second);
        std::cout << "Move: (" << bestmoveXY.first << ", " << bestmoveXY.second << ")" << std::endl;
        std::cout << "Position now: " << std::endl;
        std::cout << env << std::endl;
    }
}