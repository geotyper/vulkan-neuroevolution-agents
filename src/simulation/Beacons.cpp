#include "vkexp/simulation/Beacons.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

namespace vkexp {
namespace {

constexpr std::array<Float4, 4> colors{
    Float4{0.20F, 0.85F, 1.00F, 0.0F}, Float4{1.00F, 0.35F, 0.75F, 0.0F},
    Float4{0.55F, 1.00F, 0.35F, 0.0F}, Float4{1.00F, 0.72F, 0.20F, 0.0F}};

} // namespace

Float4 stationaryBeaconPosition(const std::uint32_t trial, const float worldRadius) {
    // Preserve the original small-world layout while scaling it with the arena.
    constexpr std::array<Float4, 4> normalizedPositions{
        Float4{1.28F / smallWorldRadius, 0.96F / smallWorldRadius, 0.0F, 0.0F},
        Float4{-1.40F / smallWorldRadius, 0.70F / smallWorldRadius, 0.0F, 0.0F},
        Float4{-0.96F / smallWorldRadius, -1.32F / smallWorldRadius, 0.0F, 0.0F},
        Float4{1.36F / smallWorldRadius, -0.92F / smallWorldRadius, 0.0F, 0.0F}};
    const Float4 normalized = normalizedPositions[trial % normalizedPositions.size()];
    return {normalized.x * worldRadius, normalized.y * worldRadius, 0.0F, 0.0F};
}

ActiveBeacons activeBeacons(const AgentState& agent, const SimulationStep& settings) {
    const std::uint32_t trial = static_cast<std::uint32_t>(std::max(agent.target.z, 0.0F));
    if (settings.beaconScenario == BeaconScenario::Stationary) {
        return {
            {{Beacon{{agent.target.x, agent.target.y, 0.0F, 0.0F}, colors[trial % colors.size()]},
              {}}},
            1};
    }

    const float offset = settings.worldRadius * 0.62F;
    const bool secondDiagonal = settings.beaconPhase != 0;
    const float firstY = secondDiagonal ? offset : -offset;
    const float secondY = -firstY;
    const std::size_t colorOffset = secondDiagonal ? 2 : 0;
    return {{{Beacon{{-offset, firstY, 0.0F, 0.0F}, colors[colorOffset]},
              Beacon{{offset, secondY, 0.0F, 0.0F}, colors[colorOffset + 1]}}},
            2};
}

float nearestBeaconDistance(const AgentState& agent, const SimulationStep& settings) {
    const ActiveBeacons beacons = activeBeacons(agent, settings);
    float nearest = settings.worldRadius * 4.0F;
    for (std::size_t index = 0; index < beacons.count; ++index) {
        const float dx = beacons.values[index].position.x - agent.pose.x;
        const float dy = beacons.values[index].position.y - agent.pose.y;
        nearest = std::min(nearest, std::sqrt(dx * dx + dy * dy));
    }
    return nearest;
}

std::uint32_t completedBeaconPhases(const AgentState& agent) {
    const auto mask = static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F) + 0.5F);
    return std::popcount(mask);
}

std::uint32_t beaconPhaseForStep(const BeaconScenario scenario, const std::uint32_t step,
                                 const std::uint32_t stepsPerGeneration) {
    if (scenario != BeaconScenario::AlternatingDiagonals || stepsPerGeneration == 0) {
        return 0;
    }
    return step >= stepsPerGeneration / 2 ? 1U : 0U;
}

} // namespace vkexp
