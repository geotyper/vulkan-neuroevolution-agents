#pragma once

#include "vkexp/simulation/AgentTypes.hpp"
#include "vkexp/worlds/ScenarioKernel.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

namespace vkexp::worlds {

// Hashing, beacon geometry and scenario timings live in ScenarioKernel.inl,
// which the shaders compile from the same source. Everything below needs C++
// types the shaders do not have.
namespace kernel = ::vkexp::worlds::kernel;

inline const float tau = kernel::ScenarioTau;
inline constexpr std::array<Float4, 4> trialColors{
    Float4{0.20F, 0.85F, 1.00F, 0.0F}, Float4{1.00F, 0.35F, 0.75F, 0.0F},
    Float4{0.55F, 1.00F, 0.35F, 0.0F}, Float4{1.00F, 0.72F, 0.20F, 0.0F}};

[[nodiscard]] inline Float4 scaledOffset(const kernel::vec2 offset, const float scale) {
    return {offset.x * scale, offset.y * scale, 0.0F, 0.0F};
}

[[nodiscard]] inline float objectiveFitness(const AgentState& agent,
                                            const std::uint32_t completedObjectives,
                                            const FitnessWeights& weights) {
    return agent.metrics.w + (agent.metrics.x - agent.metrics.y) +
           static_cast<float>(completedObjectives) * weights.objectiveBonus -
           agent.metrics.z * weights.motorCostWeight - agent.penalties.x;
}

// Banks the progress made toward the previous objective and starts measuring the
// next one. Shared by every scenario whose target can move mid-generation;
// mirrored by scenarioBankProgress in shaders/worlds/steps/shared.glsl.
inline void bankObjectiveProgress(AgentState& agent, const SimulationStep& settings,
                                  const bool clampToPositive) {
    const float progress = agent.metrics.x - agent.metrics.y;
    agent.metrics.w += clampToPositive ? std::max(progress, 0.0F) : progress;
    agent.metrics.x = nearestBeaconDistance(agent, settings);
    agent.metrics.y = agent.metrics.x;
}

// Default arrival rule: mark the active phase as completed.
inline void recordPhaseArrival(AgentState& agent, const SimulationStep& settings,
                               const float distance) {
    if (distance >= beaconArrivalRadius(settings)) {
        return;
    }
    const std::uint32_t phaseBit = 1U << settings.beaconPhase;
    const auto completedMask =
        static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F) + 0.5F) | phaseBit;
    agent.target.w = static_cast<float>(completedMask);
}

// The collect-and-deliver cycle: reach the resource, carry it home, repeat until
// the trial ends. Four scenarios run exactly this and differ only in what stands
// between the two ends, so it lives here rather than being copied a fourth time.
// Mirrored by scenarioDeliveryCycleAfterStep in shaders/worlds/steps/shared.glsl.
//
// `measureNextLegToTarget` picks what the next leg's progress is measured from.
// True uses the scenario's current target, which has just switched and is the
// thing the agent now has to reach. False uses the nearest beacon, which
// immediately after an arrival is the beacon just reached -- so the leg starts
// from zero, `metrics.y` can never fall below it, and the "got closer" gradient
// never fires for that leg at all. Forage and scent relay were written that way
// and keep it here: it is very likely a shaping bug, but changing it would
// silently change what their recorded runs mean, so it is a separate decision.
inline void deliveryCycleAfterStep(AgentState& agent, const SimulationStep& settings,
                                   const float distance, const bool measureNextLegToTarget) {
    if (agent.internal.y >= 0.5F) {
        agent.internal.x =
            std::max(0.0F, agent.internal.x - settings.forageCargoDecayRate * settings.deltaTime);
    }
    if (distance >= beaconArrivalRadius(settings)) {
        return;
    }
    agent.metrics.w += std::max(agent.metrics.x - agent.metrics.y, 0.0F);
    if (agent.internal.y >= 0.5F) {
        agent.metrics.w += agent.internal.x * settings.forageDeliveryReward;
        agent.target.w = static_cast<float>(completedForageCycles(agent) + 1);
        agent.internal.x = 0.0F;
        agent.internal.y = 0.0F;
    } else {
        agent.metrics.w += settings.foragePickupReward;
        agent.internal.x = 1.0F;
        agent.internal.y = 1.0F;
    }
    const ScenarioDefinition& scenario = scenarioDefinition(settings.beaconScenario);
    agent.metrics.x = measureNextLegToTarget && scenario.targetDistance != nullptr
                          ? scenario.targetDistance(agent, settings)
                          : nearestBeaconDistance(agent, settings);
    agent.metrics.y = agent.metrics.x;
}

// Continuous shaping for scenarios whose beacon keeps moving: without it a
// tracking agent scores nothing between arrivals.
inline void rewardVisibleTracking(AgentState& agent, const SimulationStep& settings,
                                  const float distance) {
    const float visibleCloseness =
        std::clamp(1.0F - distance / settings.lightSensorRange, 0.0F, 1.0F);
    agent.metrics.w += visibleCloseness * settings.deltaTime * settings.fitness.trackingReward;
}

} // namespace vkexp::worlds
