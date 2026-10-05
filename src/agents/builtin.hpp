#pragma once

#include "engine/actions.hpp"
#include "engine/observe.hpp"
#include <random>

namespace Agents {
    // choose uniformly from Pass and all legal full-army and half-army moves in this observation.
    Action builtin(const Observation& view, std::mt19937& random);
}
