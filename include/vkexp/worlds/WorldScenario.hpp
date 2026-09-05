#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"
#include "vkexp/worlds/ScenarioKernel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexp {

// Single source of truth shared with the shaders.
inline const float forageHomeRelocationSeconds = worlds::kernel::ForageHomeRelocationSeconds;

struct Beacon {
    Float4 position;
    Float4 color;
};

struct ActiveBeacons {
    std::array<Beacon, 2> values{};
    std::size_t count{};
};

// Beacons visible to the agent this step.
using ScenarioBeacons = ActiveBeacons (*)(const AgentState& agent, const SimulationStep& settings);
// Distance the scenario scores progress against. Null means "nearest beacon".
using ScenarioTargetDistance = float (*)(const AgentState& agent, const SimulationStep& settings);
// Active objective index for a step. Null means the scenario has a single phase.
using ScenarioPhaseForStep = std::uint32_t (*)(std::uint32_t step,
                                               std::uint32_t stepsPerGeneration);
// Final score for one finished trial, under the current shaping weights.
using ScenarioFitness = float (*)(const AgentState& agent, const FitnessWeights& weights);
// Objectives this agent actually completed, for the published arrival ratio.
using ScenarioObjectives = std::uint32_t (*)(const AgentState& agent);
// Runs before perception. Null means the scenario needs no world transition.
using ScenarioBeforeStep = void (*)(AgentState& agent, const SimulationStep& settings);
// Runs after physics with the distance to the current target.
using ScenarioAfterStep = void (*)(AgentState& agent, const SimulationStep& settings,
                                   float distance);
// Packs the scenario's own GPU parameters. `settings` must already be resolved
// for the step being packed (see resolveStepSettings).
using ScenarioGpuPacker = ScenarioParameterBlock (*)(const SimulationStep& settings);

// Which shared tunables this scenario actually reads. The UI shows controls from
// these flags instead of switching on the scenario identity.
struct ScenarioTunables {
    bool beaconRadiusRatio{};
    bool beaconAngularSpeed{};
    bool beaconRandomMotion{};
    bool forageCargoDecay{};
};

// The complete contract of one experiment. Everything the simulation, the
// scoring, the UI and the GPU need is reachable from here, so adding a scenario
// means adding one source file, one shader pair and one registry entry rather
// than extending switches across the code base.
struct ScenarioDefinition {
    const char* name{};
    // Stable short identifier for command lines and archives; unlike `name` it
    // must not change once experiments have been recorded against it.
    const char* key{};
    BeaconScenario id{};
    neuro::BrainShape brain{};
    ScenarioTunables tunables{};
    // What this scenario calls its own concepts, so the UI never has to know
    // which scenario is selected in order to label a control or a statistic.
    const char* objectiveLabel{"Beacon objectives"};
    const char* radiusLabel{"Orbit radius"};
    const char* description{};

    ScenarioBeacons beacons{};
    // How many beacons `beacons` always reports. Drawn instance count on the CPU
    // side and the loop bound on the GPU side; asserted against `beacons` in the
    // unit tests.
    std::uint32_t beaconCount{1};
    ScenarioTargetDistance targetDistance{};
    ScenarioPhaseForStep phaseForStep{};

    ScenarioFitness fitness{};
    ScenarioObjectives achievedObjectives{};
    std::uint32_t objectivesPerAgent{1};

    // Mirrored by shaders/worlds/steps/<scenario>.glsl.
    ScenarioBeforeStep beforeStep{};
    ScenarioAfterStep afterStep{};

    ScenarioGpuPacker gpuParameters{};
};

// Registry in BeaconScenario order; validated once on first use.
[[nodiscard]] std::span<const ScenarioDefinition* const> scenarioRegistry();
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
