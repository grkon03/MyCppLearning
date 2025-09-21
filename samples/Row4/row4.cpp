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

    int Row4Env::getBallcount() const { return ballcount; }
    Color Row4Env::getSquare(int x, int y, int z) const { return squares[index(x, y, z)]; }

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

    std::ostream &operator<<(std::ostream &os, const Row4Env &env)
    {
        int x, y, z;
        for (z = 3; z >= 0; --z)
        {
            os << "[z = " << z << "]" << std::endl;
            for (y = 3; y >= 0; --y)
            {
                os << "-+---+---+---+---+" << std::endl;
                os << y << "|";
                for (x = 0; x < 4; ++x)
                {
                    os << " " << env.getSquare(x, y, z) << " |";
                }
                os << std::endl;
            }
            os << "-+---+---+---+---+" << std::endl;
            os << " | 0 | 1 | 2 | 3 |" << std::endl;
        }

        return os;
    }

    Row4Agent::Row4Agent() : lengine(0.01)
    {
        using WIT = NN::Layers::AffineLayer::WeightInitType;
        using BIT = NN::Layers::AffineLayer::BiasInitType;
        pvnn.addLayer(new NN::Layers::AffineLayer(Row4Env::statesize, 100, WIT::He, BIT::SmallPositive));
        pvnn.addLayer(new NN::Layers::ReLULayer(100));
        pvnn.addLayer(new NN::Layers::AffineLayer(100, 50, WIT::He, BIT::SmallPositive));
        pvnn.addLayer(new NN::Layers::ReLULayer(50));
        auto split = new NN::Layers::SplitLayer(2);
        split->addLayer(0, new NN::Layers::AffineLayer(50, 16, WIT::He, BIT::SmallPositive));
        split->addLayer(1, new NN::Layers::AffineLayer(50, 1, WIT::He, BIT::SmallPositive));
        pvnn.addLayer(split);
        auto splitlast = new NN::Layers::SplitLastLayer(2);
        splitlast->addLastLayer(0, 1, new NN::Layers::SoftmaxLastLayer(16));
        splitlast->addLastLayer(1, 1, new NN::Layers::MSELastLayer(1));
        pvnn.setLastLayer(splitlast);
    }

    math::Real Row4Agent::update(const std::vector<RL::Transition> &transitions)
    {
        size_t size = transitions.size();
        std::vector<const math::Rmatrix *> states(size), polandval(size);
        size_t i;
        for (i = 0; i < size; ++i)
        {
            states[i] = &transitions[i].stateVector;
            polandval[i] = new math::Rmatrix(transitions[i].actionVector.connectToBottom(math::Rmatrix(transitions[i].reward)));
        }

        math::Rmatrix statescollection = math::Rmatrix::connectHorizontal(states);
        math::Rmatrix polandvalcollection = math::Rmatrix::connectHorizontal(polandval);

        pvnn.train(&lengine, statescollection, polandvalcollection);

        return pvnn.loss();
    }

    math::Rmatrix Row4Agent::policy(const RL::State &state) const
    {
        return pvnn.predict(state.vectorexp).splitRows({16})[0];
    }
    math::Real Row4Agent::value(const RL::State &state) const
    {
        return pvnn.predict(state.vectorexp).direct(16);
    }
    std::pair<math::Rmatrix, math::Real> Row4Agent::policyvalue(const RL::State &state) const
    {
        auto pv = pvnn.predict(state.vectorexp);
        return {pv.splitRows({16})[0], pv.direct(16)};
    }

    int Row4Agent::getBestAction(const Row4Env *env, const RL::Searchers::AlphaZeroMCTS *mcts) const
    {
        auto tree = mcts->initialtree(env, this);
        mcts->search(tree.root(), this);

        auto bestnode = mcts->selectByVisitCount(tree.root(), this, 0);

        return *bestnode->value().action.hotbits.begin();
    }
}