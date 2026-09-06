#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::scent_relay {

[[nodiscard]] const ScenarioDefinition& definition();
[[nodiscard]] Float4 homePosition(const AgentState& agent);
[[nodiscard]] ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] float targetDistance(const AgentState& agent, const SimulationStep& settings);

} // namespace vkexp::worlds::scent_relay
