// floats0 = {rotation angle, orbit radius ratio, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/RotatingScenario.cpp.
vec2 rotatingScenarioPosition(Agent agent, float worldRadius, ScenarioParameters sp) {
    const float orbitRadius = worldRadius * sp.floats0.y;
    const float targetLength = length(agent.target.xy);
    const vec2 base = targetLength > 0.000001
                          ? agent.target.xy / targetLength * orbitRadius
                          : vec2(orbitRadius, 0.0);
    const float cosine = cos(sp.floats0.x);
    const float sine = sin(sp.floats0.x);
    return mat2(cosine, sine, -sine, cosine) * base;
}
