#include "worlds/scenario_math.glsl"
#include "worlds/stationary.glsl"
#include "worlds/alternating.glsl"
#include "worlds/rotating.glsl"
#include "worlds/random_movement.glsl"
#include "worlds/forage_home.glsl"

uint scenarioBeaconCount(uint scenario) {
    return scenario == 1u || scenario == 4u ? 2u : 1u;
}

vec2 scenarioBeaconPosition(uint scenario, Agent agent, uint beaconIndex, uint phase,
                            float worldRadius, float motionValue, float radiusRatio,
                            float motionTime, float teleportProbability, uint motionSeed) {
    if (scenario == 0u) {
        return stationaryScenarioPosition(agent);
    }
    if (scenario == 1u) {
        return alternatingScenarioPosition(beaconIndex, phase, worldRadius);
    }
    if (scenario == 2u) {
        return rotatingScenarioPosition(agent, worldRadius, motionValue, radiusRatio);
    }
    if (scenario == 3u) {
        const uint trial = uint(max(agent.target.z, 0.0));
        return randomMovementScenarioPosition(trial, worldRadius, motionValue, radiusRatio,
                                              motionTime, teleportProbability, motionSeed);
    }
    return forageHomeScenarioPosition(agent, beaconIndex, worldRadius, motionValue,
                                      radiusRatio, motionTime, motionSeed);
}

vec3 scenarioBeaconColor(uint scenario, uint beaconIndex, uint phase, uint trial) {
    if (scenario == 1u) {
        return alternatingScenarioColor(beaconIndex, phase);
    }
    if (scenario == 4u) {
        return forageHomeScenarioColor(beaconIndex);
    }
    return stationaryScenarioColor(trial);
}

float scenarioTargetDistance(uint scenario, Agent agent, vec2 position, uint phase,
                             float worldRadius, float motionValue, float radiusRatio,
                             float motionTime, float teleportProbability, uint motionSeed) {
    if (scenario == 4u) {
        const uint targetIndex = agent.internal.y >= 0.5 ? 1u : 0u;
        return length(scenarioBeaconPosition(scenario, agent, targetIndex, phase, worldRadius,
                                             motionValue, radiusRatio, motionTime,
                                             teleportProbability, motionSeed) - position);
    }
    float nearest = worldRadius * 4.0;
    for (uint index = 0u; index < scenarioBeaconCount(scenario); ++index) {
        nearest = min(nearest,
                      length(scenarioBeaconPosition(scenario, agent, index, phase, worldRadius,
                                                    motionValue, radiusRatio, motionTime,
                                                    teleportProbability, motionSeed) - position));
    }
    return nearest;
}
