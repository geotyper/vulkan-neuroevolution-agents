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
constexpr float randomMotionSegmentSeconds = 3.0F;
constexpr float tau = 6.28318530718F;

std::uint32_t hash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

float random01(const std::uint32_t value) {
    return static_cast<float>(hash(value) & 0x00ffffffU) / 16777215.0F;
}

bool isTeleportSegment(const std::uint32_t segment, const std::uint32_t trial,
                       const SimulationStep& settings) {
    if (segment == 0 || settings.beaconTeleportProbability <= 0.0F) {
        return false;
    }
    const std::uint32_t eventKey =
        settings.beaconMotionSeed ^ (trial * 0x27d4eb2dU) ^ (segment * 0x165667b1U) ^ 0xa511e9b3U;
    return random01(eventKey) < settings.beaconTeleportProbability;
}

std::uint32_t latestWanderEpoch(const std::uint32_t segment, const std::uint32_t trial,
                                const SimulationStep& settings) {
    for (std::uint32_t candidate = segment; candidate > 0; --candidate) {
        if (isTeleportSegment(candidate, trial, settings)) {
            return candidate;
        }
    }
    return 0;
}

Float4 randomMovingBeaconPosition(const std::uint32_t trial, const SimulationStep& settings) {
    const float motionTime = std::max(settings.beaconMotionTime, 0.0F);
    const auto segment =
        static_cast<std::uint32_t>(std::floor(motionTime / randomMotionSegmentSeconds));
    const std::uint32_t epoch = latestWanderEpoch(segment, trial, settings);
    const float localTime = motionTime - static_cast<float>(epoch) * randomMotionSegmentSeconds;
    const float roamRadius = settings.worldRadius * settings.beaconRadiusRatio;
    const float scaledTime = localTime * settings.beaconRandomSpeed / std::max(roamRadius, 0.001F);
    const std::uint32_t key =
        settings.beaconMotionSeed ^ (trial * 0x9e3779b9U) ^ (epoch * 0x85ebca6bU);
    const float phase0 = random01(key) * tau;
    const float phase1 = random01(key ^ 0x68bc21ebU) * tau;
    const float phase2 = random01(key ^ 0x02e5be93U) * tau;
    const float phase3 = random01(key ^ 0x967a889bU) * tau;
    const float rawX = 0.62F * std::sin(scaledTime * 0.73F + phase0) +
                       0.28F * std::sin(scaledTime * 1.37F + phase1) +
                       0.18F * std::sin(scaledTime * 0.31F + phase2);
    const float rawY = 0.58F * std::sin(scaledTime * 0.83F + phase3) +
                       0.31F * std::sin(scaledTime * 1.19F + phase0) +
                       0.16F * std::sin(scaledTime * 0.27F + phase1);
    const float rawLength = std::hypot(rawX, rawY);
    const float scale = roamRadius / (1.25F + rawLength);
    return {rawX * scale, rawY * scale, 0.0F, 0.0F};
}

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
    if (settings.beaconScenario == BeaconScenario::Rotating) {
        const float orbitRadius = settings.worldRadius * settings.beaconRadiusRatio;
        const float targetLength = std::hypot(agent.target.x, agent.target.y);
        const float baseX =
            targetLength > 0.000001F ? agent.target.x / targetLength * orbitRadius : orbitRadius;
        const float baseY =
            targetLength > 0.000001F ? agent.target.y / targetLength * orbitRadius : 0.0F;
        const float cosine = std::cos(settings.beaconRotationAngle);
        const float sine = std::sin(settings.beaconRotationAngle);
        const float rotatedX = baseX * cosine - baseY * sine;
        const float rotatedY = baseX * sine + baseY * cosine;
        return {{{Beacon{{rotatedX, rotatedY, 0.0F, 0.0F}, colors[trial % colors.size()]}, {}}}, 1};
    }
    if (settings.beaconScenario == BeaconScenario::RandomMovement) {
        return {
            {{Beacon{randomMovingBeaconPosition(trial, settings), colors[trial % colors.size()]},
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

float beaconRotationAngleForStep(const float angularSpeed, const float deltaTime,
                                 const std::uint32_t step) {
    return std::fmod(angularSpeed * deltaTime * static_cast<float>(step), tau);
}

} // namespace vkexp
