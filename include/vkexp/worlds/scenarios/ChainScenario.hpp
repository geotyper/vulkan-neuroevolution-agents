#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::chain {

[[nodiscard]] const ScenarioDefinition& definition();

// What one step of holding `neighbours` neighbours is worth, before it is
// weighted by the step length. Shared with the shader through chain.glsl, and
// exposed here so a test can state the rule independently of where it runs.
[[nodiscard]] float stepScore(std::uint32_t neighbours, std::uint32_t rewardBand,
                              std::uint32_t crowdLimit, float crowdPenalty);

// Whether that many neighbours counts as being in the chain at all: at least
// one, and not more than the crowd limit.
[[nodiscard]] bool inBand(std::uint32_t neighbours, std::uint32_t crowdLimit);

} // namespace vkexp::worlds::chain
