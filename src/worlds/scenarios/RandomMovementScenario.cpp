#include "vkexp/worlds/scenarios/RandomMovementScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::random_movement {
namespace {

constexpr float motionSegmentSeconds = 3.0F;

float fitness(const AgentState& agent) {
    return objectiveFitness(agent, completedBeaconPhases(agent));
}

// floats0 = {wander speed, roam radius ratio, motion time, teleport probability},
// integers[0] = motion seed. Unpacked by shaders/worlds/random_movement.glsl.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{settings.beaconRandomSpeed, settings.beaconRadiusRatio, settings.beaconMotionTime,
             settings.beaconTeleportProbability},
            {},
            {settings.beaconMotionSeed, 0U, 0U, 0U}};
}

constexpr neuro::BrainShape brain{neuro::Topology::inputCount - neuro::Topology::taskInputCount -
                                      neuro::Topology::recurrentMemoryCount,
                                  neuro::Topology::hiddenCount,
                                  neuro::Topology::actuatorOutputCount};
static_assert(brain.fitsCapacity());

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

Float4 beaconPosition(const std::uint32_t trial, const SimulationStep& settings) {
    const float motionTime = std::max(settings.beaconMotionTime, 0.0F);
    const auto segment = static_cast<std::uint32_t>(std::floor(motionTime / motionSegmentSeconds));
    const std::uint32_t epoch = latestWanderEpoch(segment, trial, settings);
    const float localTime = motionTime - static_cast<float>(epoch) * motionSegmentSeconds;
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
    const float scale = roamRadius / (1.25F + std::hypot(rawX, rawY));
    return {rawX * scale, rawY * scale, 0.0F, 0.0F};
}

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{"Random movement", brain, fitness, gpuParameters};
    return value;
}

ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings) {
    const auto trial = static_cast<std::uint32_t>(std::max(agent.target.z, 0.0F));
    return {
        {{Beacon{beaconPosition(trial, settings), trialColors[trial % trialColors.size()]}, {}}},
        1};
}

} // namespace vkexp::worlds::random_movement
