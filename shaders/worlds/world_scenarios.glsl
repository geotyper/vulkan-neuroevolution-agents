#include "worlds/scenario_params.glsl"
#include "worlds/scenario_math.glsl"
#include "worlds/stationary.glsl"
#include "worlds/alternating.glsl"
#include "worlds/rotating.glsl"
#include "worlds/random_movement.glsl"
#include "worlds/forage_home.glsl"

vec2 scenarioBeaconPosition(uint scenario, Agent agent, uint beaconIndex, uint phase,
                            float worldRadius, ScenarioParameters sp) {
    if (scenario == 0u) {
        return stationaryScenarioPosition(agent);
    }
    if (scenario == 1u) {
        return alternatingScenarioPosition(beaconIndex, phase, worldRadius);
    }
    if (scenario == 2u) {
        return rotatingScenarioPosition(agent, worldRadius, sp);
    }
    if (scenario == 3u) {
        const uint trial = uint(max(agent.target.z, 0.0));
        return randomMovementScenarioPosition(trial, worldRadius, sp);
    }
    return forageHomeScenarioPosition(agent, beaconIndex, worldRadius, sp);
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

// Scenario 4 scores against the beacon the agent is currently seeking rather than
// the nearest one; the others take the nearest of `beaconCount` beacons. The
// count comes from the scenario definition through the step parameters, so this
// file never enumerates which scenarios have two beacons.
float scenarioTargetDistance(uint scenario, Agent agent, vec2 position, uint phase,
                             float worldRadius, uint beaconCount, ScenarioParameters sp) {
    if (scenario == 4u) {
        const uint targetIndex = agent.internal.y >= 0.5 ? 1u : 0u;
        return length(scenarioBeaconPosition(scenario, agent, targetIndex, phase, worldRadius,
                                             sp) - position);
    }
    float nearest = worldRadius * 4.0;
    for (uint index = 0u; index < beaconCount; ++index) {
        nearest = min(nearest,
                      length(scenarioBeaconPosition(scenario, agent, index, phase, worldRadius,
                                                    sp) - position));
    }
    return nearest;
}

// Optional scenario-specific body colouring for the visualization.
vec3 scenarioBodyTint(uint scenario, Agent agent, vec3 bodyColor) {
    if (scenario == 4u) {
        return forageHomeScenarioBodyTint(agent, bodyColor);
    }
    return bodyColor;
}
