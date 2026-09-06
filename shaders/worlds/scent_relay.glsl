// floats0 = {resource rotation angle, orbit radius ratio, unused, cargo decay rate},
// floats1 = {pickup reward, delivery reward, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/ScentRelayScenario.cpp.

float scentRelayScenarioCargoDecayRate(ScenarioParameters sp) {
    return sp.floats0.w;
}

// Opposite the trial's base beacon: fixed for the trial and derivable from state
// the agent already carries, so home costs no parameter slot.
vec2 scentRelayScenarioHomePosition(Agent agent) {
    return -agent.target.xy;
}

vec2 scentRelayScenarioPosition(Agent agent, uint beaconIndex, float worldRadius,
                                ScenarioParameters sp) {
    return beaconIndex == 0u ? rotatingScenarioPosition(agent, worldRadius, sp)
                             : scentRelayScenarioHomePosition(agent);
}

// Home is black on purpose: it emits no light for the receptors and lays no
// scent, so the return leg has nothing but the agent's own memory and marks.
vec3 scentRelayScenarioColor(uint beaconIndex) {
    return beaconIndex == 0u ? vec3(1.00, 0.82, 0.20) : vec3(0.0);
}

vec3 scentRelayScenarioBodyTint(Agent agent, vec3 bodyColor) {
    if (agent.internal.y < 0.5) {
        return bodyColor;
    }
    return mix(bodyColor, vec3(1.00, 0.82, 0.20), 0.35 + 0.45 * clamp(agent.internal.x, 0.0, 1.0));
}
