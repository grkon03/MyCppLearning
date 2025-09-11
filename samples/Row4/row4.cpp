#include "row4.hpp"

namespace MCLSamples::Row4
{
    namespace
    {
        using _line_t = std::vector<int>;
        std::vector<_line_t> __relatedLines(int index)
        {
            auto _index = Row4Env::index;
            std::vector<_line_t> lines;

            auto [x, y, z] = Row4Env::pos(index);

            bool isXMiddle = (x == 1) || (x == 2), isYMiddle = (y == 1) || (y == 2), isZMiddle = (z == 1) || (z == 2);

            // x straight

            lines.push_back({_index(0, y, z), _index(1, y, z), _index(2, y, z), _index(3, y, z)});

            // y straight

            lines.push_back({_index(x, 0, z), _index(x, 1, z), _index(x, 2, z), _index(x, 3, z)});

            // z straight

            if (z == 3)
            {
                lines.push_back({_index(x, y, 0), _index(x, y, 1), _index(x, y, 2), _index(x, y, 3)});
            }

            // xy diagonal

            if (x == y)
            {
                lines.push_back({_index(0, 0, z), _index(1, 1, z), _index(2, 2, z), _index(3, 3, z)});
            }

            if (x == 3 - y)
            {
                lines.push_back({_index(0, 3, z), _index(1, 2, z), _index(2, 1, z), _index(3, 0, z)});
            }

            // yz diagonal

            if (y == z)
            {
                lines.push_back({_index(x, 0, 0), _index(x, 1, 1), _index(x, 2, 2), _index(x, 3, 3)});
            }

            if (y == 3 - z)
            {
                lines.push_back({_index(x, 0, 3), _index(x, 1, 2), _index(x, 2, 1), _index(x, 3, 0)});
            }

            // zx diagonal

            if (z == x)
            {
                lines.push_back({_index(0, y, 0), _index(1, y, 1), _index(2, y, 2), _index(3, y, 3)});
            }

            if (z == 3 - x)
            {
                lines.push_back({_index(3, y, 0), _index(2, y, 1), _index(1, y, 2), _index(0, y, 3)});
            }

            // xyz diagonal

            if (x == y && y == z)
            {
                lines.push_back({_index(0, 0, 0), _index(1, 1, 1), _index(2, 2, 2), _index(3, 3, 3)});
            }

            if (x == y && y == 3 - z)
            {
                lines.push_back({_index(0, 0, 3), _index(1, 1, 2), _index(2, 2, 1), _index(3, 3, 0)});
            }

            if (x == 3 - y && y == z)
            {
                lines.push_back({_index(3, 0, 0), _index(2, 1, 1), _index(1, 2, 2), _index(0, 3, 3)});
            }

            if (x == 3 - y && 3 - y == z)
            {
                lines.push_back({_index(0, 3, 0), _index(1, 2, 1), _index(2, 1, 2), _index(3, 0, 3)});
            }

            return lines;
        }

        std::vector<std::vector<_line_t>> __relatedLinesForEachSquare()
        {
            std::vector<std::vector<_line_t>> linesForEachSquare(64);

            for (int i = 0; i < 64; ++i)
            {
                linesForEachSquare[i] = __relatedLines(i);
            }

            return linesForEachSquare;
        };

        std::vector<std::vector<_line_t>> _relatedLinesForEachSquare = __relatedLinesForEachSquare();
    }

    Row4Env::Row4Env()
        : squares(64, Color::Empty), sideToMove(Color::White),
          vecstate(math::Rmatrix(Row4Env::statesize, 1, 0)), winner(Color::Empty), ballcount(0) {}

    Row4Env::Row4Env(const Row4Env &env)
        : squares(env.squares), sideToMove(env.sideToMove), vecstate(env.vecstate),
          winner(env.winner), ballcount(env.ballcount) {}

    void Row4Env::write(int index, Color color)
    {
        squares[index] = color;
        switch (color)
        {
        case Color::White:
            vecstate.direct(index) = 1;
            vecstate.direct(index + 64) = 0;
            break;
        case Color::Black:
            vecstate.direct(index + 64) = 1;
            vecstate.direct(index) = 0;
            break;
        default:
            vecstate.direct(index) = 0;
            vecstate.direct(index + 64) = 0;
        }

        for (auto line : _relatedLinesForEachSquare[index])
        {
            if (
                squares[line[0]] == color &&
                squares[line[1]] == color &&
                squares[line[2]] == color &&
                squares[line[3]] == color)
            {
                winner = color;
                return;
            }
        }
    }

    void Row4Env::setSideToMove(Color color)
    {
        sideToMove = color;
        vecstate.direct(128) = colorbb(color);
    }

    Row4Env::StepReturn Row4Env::step(int x, int y)
    {
        assert(!done());

        math::Rmatrix beforestate = vecstate;
        int idx = index(x, y, 0);
        for (int z = 0; z < 4; ++z)
        {
            if (squares[idx] == Color::Empty)
            {
                ++ballcount;
                write(idx, sideToMove);
                setSideToMove(!sideToMove);
                return StepReturn{
                    .stateVector = beforestate,
                    .reward = (winner == Color::Empty) ? 0.0 : 1.0,
                    .nextStateVector = vecstate,
                    .done = done(),
                };
            }
            idx = above(idx);
        }
        assert("illegal move");

        return StepReturn{};
    }

    Row4Env::StepReturn Row4Env::step(RL::Action action)
    {
        auto [x, y] = actionpos(action.vectorexp.argmax().first);
        return step(x, y);
    }

    std::unique_ptr<RL::Environment> Row4Env::copy() const
    {
        return std::unique_ptr<RL::Environment>(new Row4Env(*this));
    }

    RL::State Row4Env::reset()
    {
        *this = Row4Env();
        return RL::State{vecstate};
    }

    RL::State Row4Env::state() const
    {
        return RL::State{vecstate};
    }

    bool Row4Env::done() const
    {
        return (winner != Color::Empty || ballcount == 64);
    }

    std::vector<RL::DiscreteAction> Row4Env::getPossibleActions() const
    {
        std::vector<RL::DiscreteAction> ret;

        int x, y;
        for (x = 0; x < 4; ++x)
        {
            for (y = 0; y < 4; ++y)
            {
                if (squares[index(x, y, 3)] == Color::Empty)
                    ret.push_back(RL::DiscreteAction{
                        .hotbits = {(size_t)actionindex(x, y)},
                        .size = 16,
                    });
            }
        }

        return ret;
    }

    Row4Agent::Row4Agent()
    {
        using WIT = NN::Layers::AffineLayer::WeightInitType;
        using BIT = NN::Layers::AffineLayer::BiasInitType;
        pvnn.addLayer(new NN::Layers::AffineLayer(Row4Env::statesize, 100, WIT::He, BIT::SmallPositive));
        pvnn.addLayer(new NN::Layers::ReLULayer(100));
        pvnn.addLayer(new NN::Layers::AffineLayer(100, 50, WIT::He, BIT::SmallPositive));
        pvnn.addLayer(new NN::Layers::ReLULayer(50));
        pvnn.addLayer(new NN::Layers::AffineLayer(50, 17, WIT::He, BIT::SmallPositive));
    }
}