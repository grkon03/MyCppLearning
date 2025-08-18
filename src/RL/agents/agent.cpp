#include "agent.hpp"

namespace MCL::RL
{
    VectorAgent::VectorAgent() : VectorAgent(0, 0) {}
    VectorAgent::VectorAgent(size_t statesize, size_t actionsize) : statesize(statesize), actionsize(actionsize) {}

    void VectorAgent::setStateSize(size_t _statesize) { statesize = _statesize; }
    void VectorAgent::setActionSize(size_t _actionsize) { actionsize = _actionsize; }
}