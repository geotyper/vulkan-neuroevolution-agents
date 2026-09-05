#pragma once

#include "vkexp/simulation/AgentTypes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace vkexp {

struct Beacon {
    Float4 position;
    Float4 color;
};

struct ActiveBeacons {
    std::array<Beacon, 2> values{};
    std::size_t count{};
};

[[nodiscard]] Float4 stationaryBeaconPosition(std::uint32_t trial, float worldRadius);
[[nodiscard]] Float4 homeBeaconPosition(const AgentState& agent, float worldRadius);
[[nodiscard]] ActiveBeacons activeBeacons(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] float nearestBeaconDistance(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] std::uint32_t completedBeaconPhases(const AgentState& agent);
[[nodiscard]] std::uint32_t completedForageCycles(const AgentState& agent);
[[nodiscard]] std::uint32_t beaconPhaseForStep(BeaconScenario scenario, std::uint32_t step,
                                               std::uint32_t stepsPerGeneration);
[[nodiscard]] float beaconRotationAngleForStep(float angularSpeed, float deltaTime,
                                               std::uint32_t step);

} // namespace vkexp
