#pragma once

#include <mcl.hpp>
#include <iostream>

using namespace MCL;

namespace MCLSamples::RL3D4R
{
    extern double __epsilon;

    enum class Color
    {
        Empty,
        White,
        Black,
    };

    inline std::ostream &operator<<(std::ostream &os, Color c)
    {
        switch (c)
        {
        case Color::Empty:
            os << " ";
            break;
        case Color::White:
            os << "W";
            break;
        case Color::Black:
            os << "B";
            break;
        }

        return os;
    }

    inline Color operator!(Color c)
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

    class Position : public RL::VectorState
    {
    public:
        using State::StateType;

    private:
        Color squares[64];

        Color turn;

        // bitboards: first 64 bits are of white, last 64 bits are of black.
        math::Rmatrix bbs;

        int ballcount;

    public:
        Position();
        Position(const VectorState *);
        Position(const Position &);

        StateType getState() const override;

        bool putball(int x, int y);
        void write(int index, Color color);
        Color at(int x, int y, int z) const;
        Color getTurn() const;
        int getBallCount() const;

        static constexpr int index(int x, int y, int z)
        {
            return (x * 16 + y * 4 + z);
        }

        static constexpr int indexAbove(int index)
        {
            return index + 1;
        }

        static constexpr size_t size = 128;

        Color winner() const;
        // x * 4 + y
        std::vector<int> legalmoves() const;
    };

    inline std::ostream &operator<<(std::ostream &os, Position pos)
    {
        int x, y, z;

        for (z = 3; z >= 0; --z)
        {
            os << "[z = " << z << "]" << std::endl;
            os << "+---------------+" << std::endl;
            for (y = 3; y >= 0; --y)
            {
                os << y;
                for (x = 0; x < 4; ++x)
                {
                    os << " " << pos.at(x, y, z) << " |";
                }
                os << std::endl;
                if (y != 0)
                {
                    os << "|---+---+---+---|" << std::endl;
                }
            }
            os << "+-0---1---2---3-+" << std::endl;
        }

        return os;
    }

    using Move = RL::VectorAction;

    class GameAgent : public RL::Agents::VNNAgent
    {
    public:
        using Agent::Action;
        using Agent::State;

    private:
        void resetNN();

    public:
        GameAgent();
        GameAgent(const RL::Agents::VNNAgent *);

        GameAgent *copy() const;
        math::Real loss() const;
        math::Real loss(Position pos, math::Real target) const;
    };

    class GameEnv : public RL::VectorEnvironment
    {
    public:
        using Environment::Action;
        using Environment::State;
        using Environment::StepReturn;

    private:
        Position position;

    public:
        GameEnv();

        StepReturn step(Action *action) override;
        const Position *state() const override;
        const Position *reset() override;
    };

    GameAgent::Action *getBestmove(const Position *, const RL::Agents::VNNAgent *);
}