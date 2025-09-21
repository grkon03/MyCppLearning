#pragma once

#include <mcl.hpp>
#include <vector>

namespace MCLSamples::Row4
{
    using namespace MCL;

    enum class Color
    {
        Empty = 0,
        White,
        Black,
    };

    inline std::ostream &operator<<(std::ostream &os, Color color)
    {
        switch (color)
        {
        case Color::White:
            os << "W";
            break;
        case Color::Black:
            os << "B";
            break;
        case Color::Empty:
            os << " ";
            break;
        default:
            assert(false);
        }
        return os;
    }

    constexpr Color operator!(Color c)
    {
        switch (c)
        {
        case Color::White:
            return Color::Black;
        case Color::Black:
            return Color::White;
        default:
            return Color::Empty;
        }
    }

    constexpr double colorbb(Color c)
    {
        switch (c)
        {
        case Color::White:
            return 1.0;
        case Color::Black:
            return -1.0;
        default:
            return 0;
        }
    }

    class Row4Env : public RL::Environments::DiscreteActionEnvironment
    {
    private:
        std::vector<Color> squares;
        Color sideToMove;
        math::Rmatrix vecstate;
        Color winner;
        int ballcount;

    public:
        constexpr static size_t statesize = 129; // while ball bbs (64) + black ball bbs (64) + color (1)

        ~Row4Env() = default;

        Row4Env();
        Row4Env(const Row4Env &);

        constexpr static std::pair<int, int> actionpos(int actionindex) { return {actionindex / 4, actionindex % 4}; }
        constexpr static std::tuple<int, int, int> pos(int index) { return {index / 16, (index / 4) % 4, index % 4}; }
        constexpr static int actionindex(int x, int y) { return (x * 4 + y); }
        constexpr static int index(int x, int y, int z) { return (x * 16 + y * 4 + z); }
        constexpr static int above(int index) { return index + 1; }

        void write(int index, Color ballcolor);
        void setSideToMove(Color color);
        int getBallcount() const;
        Color getSquare(int x, int y, int z) const;

        StepReturn step(int x, int y);
        virtual StepReturn step(RL::Action) override;
        virtual std::unique_ptr<Environment> copy() const override;
        virtual RL::State state() const override;
        virtual RL::State reset() override;
        virtual bool done() const override;

        virtual std::vector<RL::DiscreteAction> getPossibleActions() const override;
    };

    std::ostream &operator<<(std::ostream &os, const Row4Env &env);

    class Row4Agent : public RL::Agents::PVAgent
    {
    private:
        mutable NN::NeuralNetwork pvnn;
        NN::Engines::GradientDescentEngine lengine;

    public:
        Row4Agent();

        math::Real update(const std::vector<RL::Transition> &) override;
        math::Rmatrix policy(const RL::State &state) const override;
        math::Real value(const RL::State &state) const override;
        std::pair<math::Rmatrix, math::Real> policyvalue(const RL::State &state) const override;

        int getBestAction(const Row4Env *env, const RL::Searchers::AlphaZeroMCTS *mcts) const;
    };
}