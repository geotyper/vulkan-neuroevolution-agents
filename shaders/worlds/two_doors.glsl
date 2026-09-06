// A wall with two gaps, one of them a dead end. Nothing is packed: the geometry
// comes from the arena radius through the shared kernel, and which gap is
// blocked comes from the trial the agent already carries, so this file only
// says which numbers mean what.
//
// floats0 = {cargo decay rate, unused, unused, unused},
// floats1 = {pickup reward, delivery reward, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/TwoDoorsScenario.cpp.

float twoDoorsScenarioCargoDecayRate(ScenarioParameters sp) {
    return sp.floats0.x;
}

uint twoDoorsScenarioBlockedDoor(Agent agent) {
    return twoDoorsBlockedDoor(uint(max(agent.target.z, 0.0)));
}

vec2 twoDoorsScenarioPosition(uint beaconIndex, float worldRadius) {
    return beaconIndex == 0u ? twoDoorsResourcePosition(worldRadius)
                             : twoDoorsHomePosition(worldRadius);
}

// The resource is lit like any goal; home is lit too, because the question this
// world asks is which gap leads through and an invisible home would stack a
// second one on top of it.
vec3 twoDoorsScenarioColor(uint beaconIndex) {
    return beaconIndex == 0u ? vec3(1.00, 0.82, 0.20) : vec3(0.20, 0.55, 1.00);
}

vec3 twoDoorsScenarioBodyTint(Agent agent, vec3 bodyColor) {
    if (agent.internal.y < 0.5) {
        return bodyColor;
    }
    return mix(bodyColor, vec3(1.00, 0.82, 0.20), 0.35 + 0.45 * clamp(agent.internal.x, 0.0, 1.0));
}
