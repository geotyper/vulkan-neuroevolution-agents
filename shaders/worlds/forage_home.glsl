const float HomeRelocationSeconds = 8.0;

bool forageHomeScenarioRelocated(float motionTime, float deltaTime) {
    const float currentTime = max(motionTime, 0.0);
    const float previousTime = max(currentTime - deltaTime, 0.0);
    const float epsilon = max(deltaTime * 0.01, 0.000001);
    return uint(floor((currentTime + epsilon) / HomeRelocationSeconds)) !=
           uint(floor((previousTime + epsilon) / HomeRelocationSeconds));
}

vec2 forageHomeScenarioPosition(Agent agent, float worldRadius, float motionTime,
                                uint motionSeed) {
    const uint trial = uint(max(agent.target.z, 0.0));
    const uint epoch = uint(floor(max(motionTime, 0.0) / HomeRelocationSeconds));
    const uint key = motionSeed ^ (trial * 0x51ed270bu) ^
                     (epoch * 0x85ebca6bu) ^ 0xc2b2ae35u;
    const float angle = scenarioRandom01(key) * Tau;
    const float radiusRatio = 0.38 + scenarioRandom01(key ^ 0x27d4eb2du) * 0.32;
    const float radius = worldRadius * radiusRatio;
    return vec2(cos(angle), sin(angle)) * radius;
}

vec2 forageHomeScenarioPosition(Agent agent, uint beaconIndex, float worldRadius,
                                float motionValue, float radiusRatio, float motionTime,
                                uint motionSeed) {
    return beaconIndex == 0u
               ? rotatingScenarioPosition(agent, worldRadius, motionValue, radiusRatio)
               : forageHomeScenarioPosition(agent, worldRadius, motionTime, motionSeed);
}

vec3 forageHomeScenarioColor(uint beaconIndex) {
    return beaconIndex == 0u ? vec3(1.00, 0.55, 0.08) : vec3(0.12, 0.72, 1.00);
}
