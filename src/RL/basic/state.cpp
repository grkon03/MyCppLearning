#include "state.hpp"

namespace MCL::RL
{

    VectorState::VectorState() {}
    VectorState::VectorState(VectorState::StateType state) : state(state) {}
    VectorState::VectorState(const VectorState &state) : state(state.state) {}

    VectorState::StateType VectorState::getState() const { return state; }
    void VectorState::setState(VectorState::StateType _state) { state = _state; }
}