#include <mcl.hpp>
#include "3d4r.hpp"

using namespace MCLSamples::RL3D4R;
using namespace MCL;

int main()
{
    GameAgent agent;
    GameAgent beginner;
    GameEnv env;
    NN::Engines::GradientDescentEngine engine(0.001);
    RL::Runners::VectorRunner runner(&agent, &env);

    agent.setLearningEngine(&engine);
    beginner.setLearningEngine(&engine);
    agent.useTargetNetwork(128);
    beginner.useTargetNetwork(128);

    __epsilon = 0.5;
    runner.trainEpisodesMC(10000);
    std::cout << "strong agent: 10000 games learned" << std::endl;

    __epsilon = 0.1;
    engine.setRate(0.005);
    runner.trainEpisodesMC(90000);
    std::cout << "strong agent: 100000 games learned" << std::endl;

    agent.save("param.bin");

    runner.setAgent(&beginner);

    __epsilon = 0.5;
    runner.trainEpisodesMC(100);
    std::cout << "beginner agent: 100 games learned" << std::endl;

    __epsilon = 0.1;
    engine.setRate(0.005);
    runner.trainEpisodesMC(300);
    std::cout << "beginner agent: 400 games learned" << std::endl;

    env.reset();

    bool done = false;
    int x, y, mi;
    GameAgent::Action *bestmove;

    GameAgent *player;
    GameEnv::StepReturn ret;

    size_t i, experiments = 1000, strongwins = 0, beginnerwins = 0, draws = 0;

    while (!done)
    {
        switch (env.state()->getTurn())
        {
        case Color::White:
            player = &beginner;
            break;
        case Color::Black:
            player = &agent;
            break;
        default:
        }

        bestmove = getBestmove(env.state(), player);

        mi = bestmove->getAction().argmax().first;
        x = mi / 4;
        y = mi % 4;
        std::cout << env.state()->getTurn() << " move: " << x << y << std::endl;
        ret = env.step(bestmove);
        std::cout << *(env.state()) << std::endl;
        std::cout << "strong   agent eval: " << agent.evaluation(env.state()) << std::endl;
        std::cout << "beginner agent eval: " << beginner.evaluation(env.state()) << std::endl;
        done = ret.done;
    }

    std::cout << agent.loss(*env.state(), ret.reward) << std::endl;

    std::cout << env.state()->winner() << " wins this game" << std::endl;

    std::cout << std::endl
              << "play another 1000 games for each color permutation...: " << std::endl;

    // white: strong agent, black: beginner agent
    for (i = 0; i < experiments; ++i)
    {
        env.reset();
        done = false;
        while (!done)
        {
            switch (env.state()->getTurn())
            {
            case Color::White:
                player = &agent;
                break;
            case Color::Black:
                player = &beginner;
                break;
            default:
            }

            ret = env.step(player->getAction(env.state()));
            done = ret.done;
        }

        switch (env.state()->winner())
        {
        case Color::White:
            ++strongwins;
            break;
        case Color::Black:
            ++beginnerwins;
            break;
        default:
            ++draws;
        }
    }

    // white: beginner agent, black: strong agent
    for (i = 0; i < experiments; ++i)
    {
        env.reset();
        done = false;
        while (!done)
        {
            switch (env.state()->getTurn())
            {
            case Color::White:
                player = &beginner;
                break;
            case Color::Black:
                player = &agent;
                break;
            default:
            }

            ret = env.step(player->getAction(env.state()));
            done = ret.done;
        }

        switch (env.state()->winner())
        {
        case Color::White:
            ++beginnerwins;
            break;
        case Color::Black:
            ++strongwins;
            break;
        default:
            ++draws;
        }
    }

    std::cout << "strong agent wins: " << strongwins << std::endl
              << "draw: " << draws << std::endl
              << "beginner agent wins: " << beginnerwins << std::endl;
}