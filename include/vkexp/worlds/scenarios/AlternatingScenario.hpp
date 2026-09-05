#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::alternating {

[[nodiscard]] ActiveBeacons beacons(const SimulationStep& settings);
[[nodiscard]] std::uint32_t phaseForStep(std::uint32_t step,
                                         std::uint32_t stepsPerGeneration);

} // namespace vkexp::worlds::alternating
