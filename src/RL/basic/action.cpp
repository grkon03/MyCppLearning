#include "action.hpp"

namespace MCL::RL
{
    VectorAction::VectorAction() {}
    VectorAction::VectorAction(ActionType action) : action(action) {}
    VectorAction::VectorAction(const VectorAction &action) : action(action.action) {}

    VectorAction::ActionType VectorAction::getAction() const { return action; }
    void VectorAction::setAction(ActionType _action) { action = _action; }
}