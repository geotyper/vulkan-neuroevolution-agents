#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::stationary {

[[nodiscard]] Float4 beaconPosition(std::uint32_t trial, float worldRadius);
[[nodiscard]] ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings);

} // namespace vkexp::worlds::stationary
