#include "vkexp/worlds/WorldScenario.hpp"

#include "vkexp/worlds/scenarios/AlternatingScenario.hpp"
#include "vkexp/worlds/scenarios/ForageHomeScenario.hpp"
#include "vkexp/worlds/scenarios/RandomMovementScenario.hpp"
#include "vkexp/worlds/scenarios/RotatingScenario.hpp"
#include "vkexp/worlds/scenarios/StationaryScenario.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace vkexp {

const ScenarioDefinition& scenarioDefinition(const BeaconScenario scenario) {
    switch (scenario) {
    case BeaconScenario::Stationary:
        return worlds::stationary::definition();
    case BeaconScenario::AlternatingDiagonals:
        return worlds::alternating::definition();
    case BeaconScenario::Rotating:
        return worlds::rotating::definition();
    case BeaconScenario::RandomMovement:
        return worlds::random_movement::definition();
    case BeaconScenario::ForageHome:
        return worlds::forage_home::definition();
    }
    return worlds::stationary::definition();
}

Float4 stationaryBeaconPosition(const std::uint32_t trial, const float worldRadius) {
    return worlds::stationary::beaconPosition(trial, worldRadius);
}

Float4 homeBeaconPosition(const AgentState& agent, const SimulationStep& settings) {
    return worlds::forage_home::homePosition(agent, settings);
}

bool homeBeaconRelocated(const SimulationStep& settings) {
    return worlds::forage_home::homeRelocated(settings);
}

ActiveBeacons activeBeacons(const AgentState& agent, const SimulationStep& settings) {
    switch (settings.beaconScenario) {
    case BeaconScenario::Stationary:
        return worlds::stationary::beacons(agent, settings);
    case BeaconScenario::AlternatingDiagonals:
        return worlds::alternating::beacons(settings);
    case BeaconScenario::Rotating:
        return worlds::rotating::beacons(agent, settings);
    case BeaconScenario::RandomMovement:
        return worlds::random_movement::beacons(agent, settings);
    case BeaconScenario::ForageHome:
        return worlds::forage_home::beacons(agent, settings);
    }
    return worlds::stationary::beacons(agent, settings);
}

float nearestBeaconDistance(const AgentState& agent, const SimulationStep& settings) {
    if (settings.beaconScenario == BeaconScenario::ForageHome) {
        return worlds::forage_home::targetDistance(agent, settings);
    }
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

std::uint32_t completedForageCycles(const AgentState& agent) {
    return static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F) + 0.5F);
}

std::uint32_t beaconPhaseForStep(const BeaconScenario scenario, const std::uint32_t step,
                                 const std::uint32_t stepsPerGeneration) {
    return scenario == BeaconScenario::AlternatingDiagonals
               ? worlds::alternating::phaseForStep(step, stepsPerGeneration)
               : 0;
}

float beaconRotationAngleForStep(const float angularSpeed, const float deltaTime,
                                 const std::uint32_t step) {
    return worlds::rotating::angleForStep(angularSpeed, deltaTime, step);
}

SimulationStep resolveStepSettings(const SimulationStep& base, const std::uint32_t step,
                                   const std::uint32_t stepsPerGeneration) {
    SimulationStep resolved = base;
    resolved.beaconPhase = beaconPhaseForStep(base.beaconScenario, step, stepsPerGeneration);
    resolved.beaconPhaseChanged = base.beaconScenario == BeaconScenario::AlternatingDiagonals &&
                                  stepsPerGeneration != 0 && step == stepsPerGeneration / 2;
    resolved.beaconRotationAngle =
        beaconRotationAngleForStep(base.beaconAngularSpeed, base.deltaTime, step);
    resolved.beaconMotionTime = base.deltaTime * static_cast<float>(step);
    return resolved;
}

} // namespace vkexp
