#include "azmcts.hpp"
#include "../agents/policy.hpp"
#include "../environments/discrete.hpp"
#include "../util/tree.hpp"
#include "../util/random.hpp"

namespace MCL::RL::Searchers
{
    AlphaZeroMCTS::AlphaZeroMCTS() : rndgen(std::random_device()()) {}

    Episode AlphaZeroMCTS::makeEpisode(const Environment *_env, const Agent *_agent) const
    {
        const DiscreteEnv *env = Environments::cast<DiscreteEnv>(_env);
        const PVAgent *agent = Agents::cast<PVAgent>(_agent);
        Episode episode;

        Tree tree = initialtree(env, agent);

        tree.root()->value().env->reset();

        std::shared_ptr<Node> node = tree.root(), nextnode;

        while (!node->value().done)
        {
            search(node, agent);
            nextnode = selectByVisitCount(node, agent, setting.tau(node->value().env.get()));
            episode.transitions.push_back(Transition{
                .stateVector = node->value().state,
                .actionVector = Action(nextnode->value().action).vectorexp,
                .reward = nextnode->value().reward,
                .nextStateVector = nextnode->value().state,
                .done = nextnode->value().done,
            });
            node = nextnode;
        }

        math::Real rewardsum = 0;

        for (auto itr = episode.transitions.rbegin(); itr != episode.transitions.rend(); ++itr)
        {
            rewardsum = itr->reward += setting.gamma * rewardsum;
        }

        return episode;
    }

    AlphaZeroMCTS::Tree AlphaZeroMCTS::initialtree(const DiscreteEnv *env, const PVAgent *agent) const
    {
        Tree tree;
        tree.root()->value() = SearchNode{
            .env = std::unique_ptr<DiscreteEnv>(Environments::cast<DiscreteEnv>(env->copy().release())),
            .state = env->state().vectorexp,
            .reward = 0,
            .done = env->done(),
            .W = 0,
            .N = 0,
        };

        // make the prior distribution of the root node

        auto possibleActions = env->getPossibleActions();
        size_t i, noActions = possibleActions.size();
        auto policy = agent->policy(env->state());
        auto dirichlet = util::uniformDirichletSample(setting.dirichletAlpha, noActions, rndgen);
        math::Real priorsum = 0, probact;
        std::vector<math::Real> prior;

        prior = std::vector<math::Real>(noActions);

        for (i = 0; i < noActions; ++i)
        {
            probact = setting.policyInterpreter(policy, possibleActions[i]);
            prior[i] = (1 - setting.dirichletEpsilon) * probact + setting.dirichletEpsilon * dirichlet[i];
            priorsum += prior[i];
        }

        for (i = 0; i < noActions; ++i)
        {
            prior[i] /= priorsum;
        }

        tree.root()->value().P = prior;
        tree.root()->value().noActions = noActions;

        return tree;
    }

    void AlphaZeroMCTS::search(std::shared_ptr<Node> node, const PVAgent *agent) const
    {
        size_t i;
        for (i = 0; i < setting.noSimulations; ++i)
        {
            simulate(node, agent);
        }
    }

    void AlphaZeroMCTS::simulate(std::shared_ptr<Node> node, const PVAgent *agent) const
    {
        auto searching = node;
        while (searching->value().N != 0)
        {
            searching = selectByPUCT(searching, agent);
        }

        auto value = expand(searching, agent);
        backpropagate(searching, value);
    }

    std::shared_ptr<AlphaZeroMCTS::Node> AlphaZeroMCTS::selectByPUCT(std::shared_ptr<Node> node, const PVAgent *agent) const
    {
        const SearchNode &value = node->value();
        std::vector<math::Real> U(value.noActions); // UCB
        math::Real Q;
        auto children = node->children();

        size_t i = 0;
        for (i = 0; i < value.noActions; ++i)
        {
            const SearchNode &childvalue = children[i]->value();
            Q = childvalue.N == 0 ? 0 : childvalue.W / childvalue.N;
            U[i] = Q + setting.constantPUCT * value.P[i] * std::sqrt(childvalue.N) / value.N;
        }

        auto itrArgmax = std::max_element(U.begin(), U.end());
        size_t iArgmax = std::distance(U.begin(), itrArgmax);

        return children[iArgmax];
    }

    std::shared_ptr<AlphaZeroMCTS::Node> AlphaZeroMCTS::selectByVisitCount(
        std::shared_ptr<Node> node, const PVAgent *agent, math::Real tau) const
    {
        auto children = node->children();

        if (tau < setting.tauCutThreshold)
        {
            // tau == 0: argmax N
            auto itrArgmax = std::max_element(children.begin(), children.end(),
                                              [](const std::shared_ptr<Node> &lhs, const std::shared_ptr<Node> &rhs)
                                              { return lhs->value().N < rhs->value().N; });
            return *itrArgmax;
        }

        size_t i, noActions = node->value().noActions;
        std::vector<math::Real> probsWithTemperature(noActions);
        math::Real tauInv = 1 / tau;

        for (i = 0; i < noActions; ++i)
        {
            probsWithTemperature[i] = std::pow(children[i]->value().N, tauInv);
        }

        std::discrete_distribution<size_t> dist(probsWithTemperature.begin(), probsWithTemperature.end());
        return children[dist(rndgen)];
    }

    math::Real AlphaZeroMCTS::expand(std::shared_ptr<Node> node, const PVAgent *agent) const
    {
        if (node->value().done)
        {
            return node->value().reward;
        }

        auto env = node->value().env.get();
        auto actions = env->getPossibleActions();
        auto [policy, value] = agent->policyvalue(env->state());
        size_t i, noActions = actions.size();

        node->value().noActions = noActions;
        std::vector<math::Real> P(noActions);
        std::vector<std::shared_ptr<Node>> children(noActions);
        DiscreteEnv::StepReturn stepret;
        DiscreteEnv *childenv;

        for (i = 0; i < noActions; ++i)
        {
            P[i] = setting.policyInterpreter(policy, actions[i]);
            childenv = Environments::cast<DiscreteEnv>(env->copy().release());
            stepret = childenv->step(actions[i]);
            children[i] = node->newchild();
            children[i]->value().env = std::unique_ptr<DiscreteEnv>(childenv);
            children[i]->value().action = actions[i];
            children[i]->value().state = stepret.nextStateVector;
            children[i]->value().reward = stepret.reward;
            children[i]->value().done = stepret.done;
        }

        node->value().P = P;

        return value;
    }

    void AlphaZeroMCTS::backpropagate(std::shared_ptr<Node> node, math::Real value) const
    {
        if (node->value().done)
        {
            ++node->value().N;
        }

        std::shared_ptr<Node> updating = node;
        math::Real gammaPoweredByItrnum = 1;

        while (updating->parent() != nullptr)
        {
            updating = node->parent();
            ++updating->value().N;
            updating->value().W += value * gammaPoweredByItrnum;
            gammaPoweredByItrnum *= setting.gamma;
        }
    }
}