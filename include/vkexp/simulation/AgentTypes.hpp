#pragma once

#include "vkexp/worlds/ScenarioKernel.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vkexp {

enum class WorldShape : std::uint32_t {
    Circle = 0,
    Square = 1,
};

enum class WorldSize : std::uint32_t {
    Small = 0,
    Medium = 1,
    Large = 2,
};

enum class BeaconScenario : std::uint32_t {
    Stationary = 0,
    AlternatingDiagonals = 1,
    Rotating = 2,
    RandomMovement = 3,
    ForageHome = 4,
};

inline constexpr std::size_t beaconScenarioCount = 5;

inline constexpr float smallWorldRadius = 1.84F;
// Single source of truth shared with the vertex shader.
inline const float beaconVisualRadius = worlds::kernel::BeaconVisualRadius;
inline constexpr std::uint32_t minimumAgentsPerWorld = 10;

[[nodiscard]] constexpr float worldRadiusForSize(const WorldSize size) {
    switch (size) {
    case WorldSize::Small:
        return smallWorldRadius;
    case WorldSize::Medium:
        return smallWorldRadius * 1.5F;
    case WorldSize::Large:
        return smallWorldRadius * 3.0F;
    }
    return smallWorldRadius;
}

[[nodiscard]] constexpr std::uint32_t
clampAgentsPerWorld(const std::uint32_t genomeCount, const std::uint32_t requestedAgentsPerWorld) {
    if (genomeCount == 0) {
        return 0;
    }
    const std::uint32_t minimum = std::min(minimumAgentsPerWorld, genomeCount);
    return std::clamp(requestedAgentsPerWorld, minimum, genomeCount);
}

[[nodiscard]] constexpr std::uint32_t worldGroupCount(const std::uint32_t genomeCount,
                                                      const std::uint32_t agentsPerWorld) {
    const std::uint32_t clamped = clampAgentsPerWorld(genomeCount, agentsPerWorld);
    return clamped == 0 ? 0 : (genomeCount + clamped - 1) / clamped;
}

[[nodiscard]] constexpr std::uint32_t logicalWorldCount(const std::uint32_t genomeCount,
                                                        const std::uint32_t agentsPerWorld,
                                                        const std::uint32_t trialsPerGenome) {
    return worldGroupCount(genomeCount, agentsPerWorld) * trialsPerGenome;
}

[[nodiscard]] constexpr std::uint32_t logicalWorldForAgent(const std::uint32_t agentIndex,
                                                           const std::uint32_t agentsPerWorld,
                                                           const std::uint32_t trialsPerGenome) {
    if (agentsPerWorld == 0 || trialsPerGenome == 0) {
        return 0;
    }
    const std::uint32_t genome = agentIndex / trialsPerGenome;
    const std::uint32_t trial = agentIndex % trialsPerGenome;
    return (genome / agentsPerWorld) * trialsPerGenome + trial;
}

[[nodiscard]] constexpr std::uint32_t agentsInLogicalWorld(const std::uint32_t genomeCount,
                                                           const std::uint32_t agentsPerWorld,
                                                           const std::uint32_t trialsPerGenome,
                                                           const std::uint32_t worldIndex) {
    const std::uint32_t clamped = clampAgentsPerWorld(genomeCount, agentsPerWorld);
    if (clamped == 0 || trialsPerGenome == 0 ||
        worldIndex >= logicalWorldCount(genomeCount, clamped, trialsPerGenome)) {
        return 0;
    }
    const std::uint32_t group = worldIndex / trialsPerGenome;
    const std::uint32_t firstGenome = group * clamped;
    return std::min(clamped, genomeCount - firstGenome);
}

struct alignas(16) Float4 {
    float x{};
    float y{};
    float z{};
    float w{};
};

// std430-compatible data shared verbatim with compute and vertex shaders.
struct alignas(16) AgentState {
    Float4 pose;        // position.xy, angle, circular collision radius
    Float4 motion;      // velocity.xy, angular velocity, normalized energy
    Float4 signal;      // emitted RGB and intensity
    Float4 target;      // base beacon.xy, trial id, completed mask or forage-cycle count
    Float4 metrics;     // phase start/min distance, motor cost, completed-phase progress
    Float4 penalties;   // wall/agent/hazard penalties and logical world id
    Float4 internal;    // cargo level, seeking-home flag, and two recurrent memory cells
    Float4 wallTouch0;  // wall contact sectors 0..3
    Float4 wallTouch1;  // wall contact sectors 4..7
    Float4 agentTouch0; // agent contact sectors 0..3
    Float4 agentTouch1; // agent contact sectors 4..7
};

static_assert(std::is_trivially_copyable_v<AgentState>);
static_assert(sizeof(AgentState) == 176);
static_assert(offsetof(AgentState, metrics) == 64);
static_assert(offsetof(AgentState, penalties) == 80);
static_assert(offsetof(AgentState, internal) == 96);
static_assert(offsetof(AgentState, wallTouch0) == 112);

// Shaping and energy coefficients. They used to be literals split between
// CpuSimulation.cpp and agent_step.comp, which made every fitness experiment a
// two-language edit plus a parity re-check; as parameters they are sliders.
struct FitnessWeights {
    float objectiveBonus{2.0F};    // score per completed objective
    float motorCostWeight{0.002F}; // fitness charged per unit of motor and signal effort
    float signalCostFactor{0.25F}; // cost of emitting, relative to moving
    float energyDrain{0.0008F};    // battery drained per unit of effort
    float trackingReward{0.25F};   // shaping for scenarios whose beacon keeps moving
};

struct SimulationStep {
    float deltaTime{1.0F / 60.0F};
    float worldRadius{smallWorldRadius};
    float thrust{1.9F};
    float turnAcceleration{5.0F};
    float linearDrag{1.7F};
    float angularDrag{2.4F};
    float sensorFieldOfView{1.8F};
    float arrivalRadiusMultiplier{1.0F};
    float maximumSpeed{0.55F};
    float maximumAngularSpeed{3.0F};
    float lightSensorRange{2.4F};
    float lightExposure{1.25F};
    float collisionRestitution{0.35F};
    float collisionStiffness{0.85F};
    float wallCollisionPenalty{0.01F};
    float beaconAngularSpeed{0.35F};
    float beaconRotationAngle{};
    float beaconRadiusRatio{0.72F};
    float beaconMotionTime{};
    float beaconTeleportProbability{0.25F};
    float beaconRandomSpeed{0.18F};
    float forageCargoDecayRate{0.08F};
    float foragePickupReward{0.25F};
    float forageDeliveryReward{4.0F};
    FitnessWeights fitness{};
    std::uint32_t beaconMotionSeed{};
    WorldShape worldShape{WorldShape::Circle};
    WorldSize worldSize{WorldSize::Small};
    BeaconScenario beaconScenario{BeaconScenario::Stationary};
    std::uint32_t beaconPhase{};
    bool beaconPhaseChanged{};
    bool agentCollisionsEnabled{true};
    bool agentLightEnabled{true};
};

[[nodiscard]] inline float beaconArrivalRadius(const SimulationStep& settings) {
    return beaconVisualRadius * settings.arrivalRadiusMultiplier;
}

// Opaque per-scenario transport. Each scenario packs and unpacks this block
// itself on both sides, so adding a scenario never widens the shared structs.
// Mirrors `ScenarioParameters` in shaders/worlds/scenario_params.glsl.
struct alignas(16) ScenarioParameterBlock {
    Float4 floats0;
    Float4 floats1;
    std::array<std::uint32_t, 4> integers{};
};

static_assert(sizeof(ScenarioParameterBlock) == 48);
static_assert(offsetof(ScenarioParameterBlock, integers) == 32);

// std430 mirror of FitnessWeights; padded so the scenario block stays aligned.
struct alignas(16) GpuFitnessWeights {
    float objectiveBonus{};
    float motorCostWeight{};
    float signalCostFactor{};
    float energyDrain{};
    float trackingReward{};
    float reserved0{};
    float reserved1{};
    float reserved2{};
};

static_assert(sizeof(GpuFitnessWeights) == 32);

// std430-compatible per-step parameters. This lives in a storage buffer rather
// than push constants: the scenario block already pushes the structure past the
// 128-byte size Vulkan guarantees for maxPushConstantsSize, and one batched
// upload per frame costs less than one vkCmdPushConstants per step.
struct alignas(16) GpuStepParameters {
    float deltaTime{};
    float worldRadius{};
    float thrust{};
    float turnAcceleration{};
    float linearDrag{};
    float angularDrag{};
    float sensorFieldOfView{};
    float arrivalRadius{};
    float maximumSpeed{};
    float maximumAngularSpeed{};
    float lightSensorRange{};
    float lightExposure{};
    float collisionRestitution{};
    float collisionStiffness{};
    float gridCellSize{};
    float wallCollisionPenalty{};
    std::uint32_t agentCount{};
    std::uint32_t brainLayout{}; // packed genome stride and active input/hidden/output counts
    std::uint32_t trialsPerGenome{};
    std::uint32_t worldShape{};
    std::uint32_t gridWidth{};
    std::uint32_t gridCellsPerWorld{};
    std::uint32_t agentCollisionsEnabled{};
    std::uint32_t agentLightEnabled{};
    std::uint32_t beaconScenario{};
    std::uint32_t beaconPhase{};
    std::uint32_t beaconPhaseChanged{};
    std::uint32_t beaconCount{};
    GpuFitnessWeights fitness;
    ScenarioParameterBlock scenario;
};

static_assert(sizeof(GpuStepParameters) == 192);
static_assert(offsetof(GpuStepParameters, agentCount) == 64);
static_assert(offsetof(GpuStepParameters, beaconScenario) == 96);
static_assert(offsetof(GpuStepParameters, fitness) == 112);
static_assert(offsetof(GpuStepParameters, scenario) == 144);

[[nodiscard]] constexpr GpuFitnessWeights packFitnessWeights(const FitnessWeights& weights) {
    return {weights.objectiveBonus,
            weights.motorCostWeight,
            weights.signalCostFactor,
            weights.energyDrain,
            weights.trackingReward,
            0.0F,
            0.0F,
            0.0F};
}

} // namespace vkexp
