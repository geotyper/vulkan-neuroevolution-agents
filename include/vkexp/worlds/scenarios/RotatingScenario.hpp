#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::rotating {

[[nodiscard]] Float4 beaconPosition(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] float angleForStep(float angularSpeed, float deltaTime, std::uint32_t step);

} // namespace vkexp::worlds::rotating
