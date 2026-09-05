vec2 rotatingScenarioPosition(Agent agent, float worldRadius, float motionValue,
                              float radiusRatio) {
    const float orbitRadius = worldRadius * radiusRatio;
    const float targetLength = length(agent.target.xy);
    const vec2 base = targetLength > 0.000001
                          ? agent.target.xy / targetLength * orbitRadius
                          : vec2(orbitRadius, 0.0);
    const float cosine = cos(motionValue);
    const float sine = sin(motionValue);
    return mat2(cosine, sine, -sine, cosine) * base;
}
