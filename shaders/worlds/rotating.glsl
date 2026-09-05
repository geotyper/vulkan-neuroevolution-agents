// floats0 = {rotation angle, orbit radius ratio, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/RotatingScenario.cpp.
vec2 rotatingScenarioPosition(Agent agent, float worldRadius, ScenarioParameters sp) {
    const float orbitRadius = worldRadius * sp.floats0.y;
    const float targetLength = length(agent.target.xy);
    const vec2 baseDirection =
        targetLength > 0.000001 ? agent.target.xy / targetLength : vec2(1.0, 0.0);
    return rotatingOrbitOffset(baseDirection, sp.floats0.x) * orbitRadius;
}
