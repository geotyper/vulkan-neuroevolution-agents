#pragma once

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
};

inline constexpr float smallWorldRadius = 1.84F;

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
    Float4 target;      // stationary beacon.xy, trial id, completed-phase bit mask
    Float4 metrics;     // phase start/min distance, motor cost, completed-phase progress
    Float4 wallTouch0;  // wall contact sectors 0..3
    Float4 wallTouch1;  // wall contact sectors 4..7
    Float4 agentTouch0; // agent contact sectors 0..3
    Float4 agentTouch1; // agent contact sectors 4..7
};

static_assert(std::is_trivially_copyable_v<AgentState>);
static_assert(sizeof(AgentState) == 144);
static_assert(offsetof(AgentState, metrics) == 64);
static_assert(offsetof(AgentState, wallTouch0) == 80);

struct SimulationStep {
    float deltaTime{1.0F / 60.0F};
    float worldRadius{smallWorldRadius};
    float thrust{1.9F};
    float turnAcceleration{5.0F};
    float linearDrag{1.7F};
    float angularDrag{2.4F};
    float sensorFieldOfView{1.8F};
    float arrivalRadius{0.055F};
    float maximumSpeed{0.55F};
    float maximumAngularSpeed{3.0F};
    float lightSensorRange{2.4F};
    float lightExposure{1.25F};
    float collisionRestitution{0.35F};
    float collisionStiffness{0.85F};
    WorldShape worldShape{WorldShape::Circle};
    WorldSize worldSize{WorldSize::Small};
    BeaconScenario beaconScenario{BeaconScenario::Stationary};
    std::uint32_t beaconPhase{};
    bool beaconPhaseChanged{};
    bool agentCollisionsEnabled{true};
    bool agentLightEnabled{true};
};

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
    float reservedFloat{};
    std::uint32_t agentCount{};
    std::uint32_t weightsPerGenome{};
    std::uint32_t trialsPerGenome{};
    std::uint32_t worldShape{};
    std::uint32_t gridWidth{};
    std::uint32_t gridCellsPerTrial{};
    std::uint32_t agentCollisionsEnabled{};
    std::uint32_t agentLightEnabled{};
    std::uint32_t beaconScenario{};
    std::uint32_t beaconPhase{};
    std::uint32_t beaconPhaseChanged{};
    std::uint32_t reservedUint{};
};

static_assert(sizeof(GpuStepParameters) == 112);

} // namespace vkexp
