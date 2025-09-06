#pragma once

#include "searcher.hpp"
#include "../environments/discrete.hpp"
#include "../agents/pv.hpp"
#include "../util/tree.hpp"
#include "../interpret/action.hpp"
#include <functional>
#include <random>

namespace MCL::RL::Searchers
{
    /**
     * @brief Monte Carlo Tree Search
     *
     * constructions: see "Mastering Chess and Shogi by Self-Play with a
     * General Reinforcement Learning Algorithm" https://arxiv.org/pdf/1712.01815
     *
     * @note Only for discrete action environments
     * @note Only for agents with policies and values
     *
     */
    class AlphaZeroMCTS : public Searcher
    {
    public:
        using DiscreteEnv = Environments::DiscreteActionEnvironment;
        using PVAgent = Agents::PVAgent;

        struct SearchNode
        {
            // environment
            std::unique_ptr<DiscreteEnv> env;
            // action
            DiscreteAction action;
            // state
            math::Rmatrix state;
            // reward which agents can get when they visit this node(this state)
            math::Real reward;
            // done by stepping action above
            bool done = false;
            // average or max of reward
            math::Real W = 0;
            // times selected
            size_t N = 0;
            // prior policy
            std::vector<math::Real> P;
            // the number of actions
            size_t noActions;
        };

        using Tree = util::Tree<SearchNode>;
        using Node = Tree::Node;

        struct Setting
        {
            math::Real constantPUCT;
            size_t noSimulations;

            /**
             * @brief
             * if the agent get reward r at k-th transition,
             * then gamma^(k-l) will be added to the reward of l-th transition for each l < k
             *
             * for boardgame, gamma should be -1 (+- epsilon)
             */
            math::Real gamma = 0;

            math::Real dirichletAlpha = 0.2;
            math::Real dirichletEpsilon = 0.2;

            /**
             * @brief tau (temperature) function
             *
             * tau is a function receiving the environment(state) and
             * return how diversely should we choose actions in that state
             *
             * tau == 0: means that actions will be deterministically chosen so that such the actions are most visited one in each state.
             * tau == 1: means that actions will be probabilistically chosen so that the distributions are
             *
             * default value: always tau == 1
             *
             */
            std::function<math::Real(const DiscreteEnv *)> tau =
                [](auto)
            { return 1; };

            /**
             * @brief if tau (temperature) is below this value, then tau will be reinterpret as 0
             *
             */
            math::Real tauCutThreshold = 0.0001;

            std::function<math::Real(math::Rmatrix, DiscreteAction)> policyInterpreter =
                interpret::action::probabilityOfActionWithPolicy;
        };

        Setting setting;

        mutable std::mt19937 rndgen;

    public:
        AlphaZeroMCTS();

        virtual Episode makeEpisode(const Environment *, const Agent *) const override;
        /**
         * @brief make new tree with a root node with the environment
         *
         * @return Tree
         */
        virtual Tree initialtree(const DiscreteEnv *, const PVAgent *) const;
        /**
         * @brief simulate the node setting.noSimulations times
         *
         */
        virtual void search(std::shared_ptr<Node> node, const PVAgent *agent) const;
        /**
         * @brief repeat selecting and expand the last node when finding a first visited node and backpropagate
         *
         */
        virtual void simulate(std::shared_ptr<Node> node, const PVAgent *agent) const;
        /**
         * @brief select node by PUCT
         *
         */
        virtual std::shared_ptr<Node> selectByPUCT(std::shared_ptr<Node> node, const PVAgent *agent) const;
        /**
         * @brief select node by visit count and tau (temperature)
         *
         */
        virtual std::shared_ptr<Node> selectByVisitCount(std::shared_ptr<Node> node, const PVAgent *agent, math::Real tau) const;
        /**
         * @brief expand the node and return the value of state
         *
         */
        virtual math::Real expand(std::shared_ptr<Node> node, const PVAgent *agent) const;
        /**
         * @brief backpropagate from the node
         *
         */
        virtual void backpropagate(std::shared_ptr<Node> node, math::Real value) const;
    };
}