#pragma once

#include "../basic/basic.hpp"
#include "../environments/environment.hpp"
#include "../agents/agent.hpp"

namespace MCL::RL
{
    class Searcher
    {
    public:
        virtual Episode makeEpisode(const Environment *, const Agent *) const = 0;
    };
}