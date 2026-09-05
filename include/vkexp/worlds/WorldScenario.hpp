#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace vkexp {

inline constexpr float forageHomeRelocationSeconds = 8.0F;

struct Beacon {
    Float4 position;
    Float4 color;
};

struct ActiveBeacons {
    std::array<Beacon, 2> values{};
    std::size_t count{};
};

using ScenarioFitness = float (*)(const AgentState& agent);

// Packs the scenario's own GPU parameters. `settings` must already be resolved
// for the step being packed (see resolveStepSettings).
using ScenarioGpuPacker = ScenarioParameterBlock (*)(const SimulationStep& settings);

struct ScenarioDefinition {
    const char* name{};
    neuro::BrainShape brain{};
    ScenarioFitness fitness{};
    ScenarioGpuPacker gpuParameters{};
};

[[nodiscard]] const ScenarioDefinition& scenarioDefinition(BeaconScenario scenario);

// Resolves the step-dependent fields (beacon phase, rotation angle, motion time)
// for `step` within a generation. Simulation and visualization share this so the
// rendered beacon cannot drift away from the simulated one.
[[nodiscard]] SimulationStep resolveStepSettings(const SimulationStep& base, std::uint32_t step,
                                                 std::uint32_t stepsPerGeneration);

[[nodiscard]] Float4 stationaryBeaconPosition(std::uint32_t trial, float worldRadius);
[[nodiscard]] Float4 homeBeaconPosition(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] bool homeBeaconRelocated(const SimulationStep& settings);
[[nodiscard]] ActiveBeacons activeBeacons(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] float nearestBeaconDistance(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] std::uint32_t completedBeaconPhases(const AgentState& agent);
[[nodiscard]] std::uint32_t completedForageCycles(const AgentState& agent);
[[nodiscard]] std::uint32_t beaconPhaseForStep(BeaconScenario scenario, std::uint32_t step,
                                               std::uint32_t stepsPerGeneration);
[[nodiscard]] float beaconRotationAngleForStep(float angularSpeed, float deltaTime,
                                               std::uint32_t step);

} // namespace vkexp
