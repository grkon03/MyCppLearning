#include <vector>
#include <random>
#include "3d4r.hpp"

namespace MCLSamples::RL3D4R
{

    double __epsilon = 0.1;

    namespace
    {
        using __line_t = std::array<int, 4>;
        using __lines_t = std::vector<__line_t>;

        __lines_t __straightlines_()
        {
            __lines_t ret;
            __line_t line;

            int i, j, k;

            for (i = 0; i < 4; ++i)
            {
                for (j = 0; j < 4; ++j)
                {
                    for (k = 0; k < 4; ++k)
                        line[k] = Position::index(i, j, k);
                    ret.push_back(line);
                    for (k = 0; k < 4; ++k)
                        line[k] = Position::index(j, k, i);
                    ret.push_back(line);
                    for (k = 0; k < 4; ++k)
                        line[k] = Position::index(k, i, j);
                    ret.push_back(line);
                }
            }

            for (i = 0; i < 4; ++i)
            {
                for (j = 0; j < 4; ++j)
                    line[j] = Position::index(i, j, j);
                ret.push_back(line);
                for (j = 0; j < 4; ++j)
                    line[j] = Position::index(i, 3 - j, j);
                ret.push_back(line);
                for (j = 0; j < 4; ++j)
                    line[j] = Position::index(j, i, j);
                ret.push_back(line);
                for (j = 0; j < 4; ++j)
                    line[j] = Position::index(3 - j, i, j);
                ret.push_back(line);
                for (j = 0; j < 4; ++j)
                    line[j] = Position::index(j, j, i);
                ret.push_back(line);
                for (j = 0; j < 4; ++j)
                    line[j] = Position::index(3 - j, j, i);
                ret.push_back(line);
            }

            line = {Position::index(0, 0, 0), Position::index(1, 1, 1), Position::index(2, 2, 2), Position::index(3, 3, 3)};
            ret.push_back(line);
            line = {Position::index(0, 0, 3), Position::index(1, 1, 2), Position::index(2, 2, 1), Position::index(3, 3, 0)};
            ret.push_back(line);
            line = {Position::index(0, 3, 0), Position::index(1, 2, 1), Position::index(2, 1, 2), Position::index(3, 0, 3)};
            ret.push_back(line);
            line = {Position::index(3, 0, 0), Position::index(2, 1, 1), Position::index(1, 2, 2), Position::index(0, 3, 3)};
            ret.push_back(line);

            return ret;
        }

        const __lines_t __straightlines = __straightlines_();

        auto __rndmtx = util::randomMatrixFromNormalDistribution;

        std::mt19937 __rndgen = std::mt19937(std::random_device()());
    }

    Position::Position() : turn(Color::White), bbs(128, 1, 0), ballcount(0)
    {
        int i;
        for (i = 0; i < 64; ++i)
        {
            squares[i] = Color::Empty;
        }
    }

    Position::Position(const Position &pos) : turn(pos.turn), bbs(pos.bbs), ballcount(pos.ballcount)
    {
        int i;
        for (i = 0; i < 64; ++i)
        {
            squares[i] = pos.squares[i];
        }
    }

    Position::StateType Position::getState() const
    {
        return bbs;
    }

    bool Position::putball(int x, int y)
    {
        Color c;
        int index, i;
        index = Position::index(x, y, 0);

        for (i = 0; i < 4; ++i)
        {
            c = squares[index];

            if (c == Color::Empty)
            {
                write(index, turn);
                turn = !turn;
                ++ballcount;
                return true;
            }

            index = Position::indexAbove(index);
        }

        return false;
    }

    void Position::write(int index, Color color)
    {
        squares[index] = color;
        switch (color)
        {
        case Color::White:
            bbs.direct(index) = 1;
            bbs.direct(index + 64) = 0;
            break;
        case Color::Black:
            bbs.direct(index) = 0;
            bbs.direct(index + 64) = 1;
            break;
        case Color::Empty:
            bbs.direct(index) = bbs.direct(index + 64) = 0;
        }
    }

    Color Position::at(int x, int y, int z) const
    {
        return squares[Position::index(x, y, z)];
    }

    Color Position::getTurn() const { return turn; }

    int Position::getBallCount() const { return ballcount; }

    Color Position::winner() const
    {
        bool wwin = false, bwin = false;
        for (auto line : __straightlines)
        {
            if (bbs.direct(line[0]) * bbs.direct(line[1]) * bbs.direct(line[2]) * bbs.direct(line[3]) != 0)
                wwin = true;
            if (bbs.direct(line[0] + 64) * bbs.direct(line[1] + 64) * bbs.direct(line[2] + 64) * bbs.direct(line[3] + 64) != 0)
                bwin = true;
        }

        if (wwin && bwin)
        {
            std::cout << *this << std::endl;
            assert(false);
        };
        if (wwin)
            return Color::White;
        else if (bwin)
            return Color::Black;
        else
            return Color::Empty;
    }

    std::vector<int> Position::legalmoves() const
    {
        std::vector<int> ret;
        int x, y;
        for (x = 0; x < 4; ++x)
        {
            for (y = 0; y < 4; ++y)
            {
                if (squares[Position::index(x, y, 3)] == Color::Empty)
                    ret.push_back(x * 4 + y);
            }
        }

        return ret;
    }

    GameAgent::GameAgent() : VNNAgent(128, 16, 0.99)
    {
        resetNN();
        this->setActionSelector(actionSelector);
    }

    GameAgent::GameAgent(const RL::Agents::VNNAgent *agent) : VNNAgent(*agent) {}

    void GameAgent::resetNN()
    {
        vfuncNN->addLayer(new NN::AffineLayer(__rndmtx(100, Position::size, 0, 0.125), math::Rmatrix(100, 1, 0)));
        vfuncNN->addLayer(new NN::ReLULayer(100));
        vfuncNN->addLayer(new NN::AffineLayer(__rndmtx(50, 100, 0, 0.14142135623), math::Rmatrix(50, 1, 0)));
        vfuncNN->addLayer(new NN::ReLULayer(50));
        vfuncNN->addLayer(new NN::AffineLayer(__rndmtx(1, 50, 0, 0.2), math::Rmatrix(1, 1, 0)));
        vfuncNN->setLastLayer(new NN::MSELastLayer(1));

        synchronizeTarget();
    }

    GameAgent *GameAgent::copy() const
    {
        return new GameAgent(VNNAgent::copy());
    }

    math::Real GameAgent::loss() const { return vfuncNN->loss(); }

    math::Real GameAgent::loss(Position pos, math::Real target) const
    {
        vfuncNN->predict(pos.getState());
        vfuncNN->learn(engine, math::Rmatrix(target));

        return vfuncNN->loss();
    }

    GameEnv::GameEnv() : RL::VectorEnvironment(), position() {}
    GameEnv::StepReturn GameEnv::step(Action *action)
    {
        auto i = action->getAction().argmax().first;
        int x = i / 4, y = i % 4;
        position.putball(x, y);
        StepReturn ret;
        switch (position.winner())
        {
        case Color::White:
            ret.done = true;
            ret.reward = 1;
            break;
        case Color::Black:
            ret.done = true;
            ret.reward = -1;
            break;
        case Color::Empty:
            ret.done = false;
            ret.reward = 0;
            break;
        }

        if (position.getBallCount() == 64)
            ret.done = true;

        return ret;
    }

    const Position *GameEnv::state() const { return &position; }
    const Position *GameEnv::reset()
    {
        position = Position();
        return &position;
    }

    GameAgent::Action *getBestmove(const Position *p, const RL::Agents::VNNAgent *agent)
    {
        math::Rmatrix actvec = math::Rmatrix(16, 1, 0);
        auto legalmoves = p->legalmoves();
        double maxeval = -1e10, eval;
        int bestmove, sign = p->getTurn() == Color::White ? 1 : -1;
        Position nextp;

        for (auto move : legalmoves)
        {
            nextp = Position(*p);
            nextp.putball(move / 4, move % 4);
            eval = sign * agent->evaluation(&nextp);
            if (maxeval < eval)
            {
                bestmove = move;
                maxeval = eval;
            }
        }
        actvec.direct(bestmove) = 1;

        return new RL::VectorAction(actvec);
    }

    // epsilon-greedy
    GameAgent::Action *actionSelector(const GameAgent::State *_s, const RL::Agents::VNNAgent *agent)
    {
        int x, y, z = 3;
        auto s = dynamic_cast<const Position *>(_s);
        std::uniform_real_distribution<double> unifdist01(0, 1);

        if (unifdist01(__rndgen) > __epsilon)
        {
            // greedy
            return getBestmove(s, agent);
        }

        std::vector<int> legalmoves = s->legalmoves();
        math::Rmatrix actvec(16, 1, 0);
        std::uniform_int_distribution<size_t> legaldist(0, legalmoves.size() - 1);

        actvec.direct(legalmoves[legaldist(__rndgen)]) = 1;

        return new RL::VectorAction(actvec);
    }
}