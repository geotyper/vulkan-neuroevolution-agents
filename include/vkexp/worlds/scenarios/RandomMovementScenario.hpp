#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::random_movement {

[[nodiscard]] const ScenarioDefinition& definition();
[[nodiscard]] ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings);

} // namespace vkexp::worlds::random_movement
