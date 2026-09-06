// Two fixed beacons with a short wall between them. Nothing is packed for the
// geometry: it comes from the arena radius through the shared kernel.
//
// floats0 = {unused, unused, unused, cargo decay rate},
// floats1 = {pickup reward, delivery reward, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/ShuttleScenario.cpp.

vec2 shuttleScenarioPosition(uint beaconIndex, float worldRadius) {
    return beaconIndex == 0u ? shuttleResourcePosition(worldRadius)
                             : shuttleHomePosition(worldRadius);
}

vec3 shuttleScenarioColor(uint beaconIndex) {
    return beaconIndex == 0u ? vec3(1.00, 0.82, 0.20) : vec3(0.20, 0.55, 1.00);
}

vec3 shuttleScenarioBodyTint(Agent agent, vec3 bodyColor) {
    if (agent.internal.y < 0.5) {
        return bodyColor;
    }
    return mix(bodyColor, vec3(1.00, 0.82, 0.20), 0.35 + 0.45 * clamp(agent.internal.x, 0.0, 1.0));
}
