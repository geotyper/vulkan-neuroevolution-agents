// floats0 = {resource rotation angle, orbit radius ratio, motion time, cargo decay rate},
// integers[0] = motion seed.
// Packed by gpuParameters in src/worlds/scenarios/ForageHomeScenario.cpp.
// Geometry and timing from the shared kernel; see
// include/vkexp/worlds/ScenarioKernel.inl.
float forageHomeScenarioCargoDecayRate(ScenarioParameters sp) {
    return sp.floats0.w;
}

bool forageHomeScenarioRelocated(ScenarioParameters sp, float deltaTime) {
    return forageHomeRelocated(sp.floats0.z, deltaTime);
}

vec2 forageHomeScenarioHomePosition(Agent agent, float worldRadius, ScenarioParameters sp) {
    const uint trial = uint(max(agent.target.z, 0.0));
    const uint key = forageHomeKey(sp.integers.x, trial, forageHomeEpoch(sp.floats0.z));
    return forageHomeOffset(key) * worldRadius;
}

vec2 forageHomeScenarioPosition(Agent agent, uint beaconIndex, float worldRadius,
                                ScenarioParameters sp) {
    return beaconIndex == 0u
               ? rotatingScenarioPosition(agent, worldRadius, sp)
               : forageHomeScenarioHomePosition(agent, worldRadius, sp);
}

vec3 forageHomeScenarioColor(uint beaconIndex) {
    return beaconIndex == 0u ? vec3(1.00, 0.55, 0.08) : vec3(0.12, 0.72, 1.00);
}

// Carrying agents are tinted toward the resource colour so a foraging cycle is
// visible at a glance.
vec3 forageHomeScenarioBodyTint(Agent agent, vec3 bodyColor) {
    if (agent.internal.y < 0.5) {
        return bodyColor;
    }
    return mix(bodyColor, vec3(1.00, 0.55, 0.08), 0.35 + 0.45 * clamp(agent.internal.x, 0.0, 1.0));
}
