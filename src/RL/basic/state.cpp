#include "state.hpp"

namespace MCL::RL
{

    VectorState::VectorState() {}

    VectorState::StateType VectorState::getState() const { return math::Rmatrix(); }
}