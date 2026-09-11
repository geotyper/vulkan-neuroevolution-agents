#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::chain {

[[nodiscard]] const ScenarioDefinition& definition();

// What one step of holding `neighbours` neighbours is worth, before it is
// weighted by the step length. Shared with the shader through chain.glsl, and
// exposed here so a test can state the rule independently of where it runs.
//
// `lopsidedness` is the magnitude of the mean unit vector to those neighbours:
// 0 when they balance, which is the middle of a chain, and 1 when they all lie
// one way, which is what a single neighbour always looks like.
// `alignment` is how far those neighbours are going the same way, on 0..1: 1 for
// a column, 0 for an orbiting pair, 0.5 for a crowd with no common direction.
[[nodiscard]] float stepScore(std::uint32_t neighbours, float lopsidedness, float alignment,
                              std::uint32_t rewardBand, std::uint32_t crowdLimit,
                              float crowdPenalty, float straightWeight, float alignWeight);

// Whether that many neighbours counts as being in the chain at all: at least
// one, and not more than the crowd limit.
[[nodiscard]] bool inBand(std::uint32_t neighbours, std::uint32_t crowdLimit);

} // namespace vkexp::worlds::chain
