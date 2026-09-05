// floats0 = {resource rotation angle, orbit radius ratio, motion time, cargo decay rate},
// integers[0] = motion seed.
// Packed by gpuParameters in src/worlds/scenarios/ForageHomeScenario.cpp.
const float HomeRelocationSeconds = 8.0;

float forageHomeScenarioCargoDecayRate(ScenarioParameters sp) {
    return sp.floats0.w;
}

bool forageHomeScenarioRelocated(ScenarioParameters sp, float deltaTime) {
    const float currentTime = max(sp.floats0.z, 0.0);
    const float previousTime = max(currentTime - deltaTime, 0.0);
    const float epsilon = max(deltaTime * 0.01, 0.000001);
    return uint(floor((currentTime + epsilon) / HomeRelocationSeconds)) !=
           uint(floor((previousTime + epsilon) / HomeRelocationSeconds));
}

vec2 forageHomeScenarioHomePosition(Agent agent, float worldRadius, ScenarioParameters sp) {
    const uint trial = uint(max(agent.target.z, 0.0));
    const uint epoch = uint(floor(max(sp.floats0.z, 0.0) / HomeRelocationSeconds));
    const uint key = sp.integers.x ^ (trial * 0x51ed270bu) ^
                     (epoch * 0x85ebca6bu) ^ 0xc2b2ae35u;
    const float angle = scenarioRandom01(key) * Tau;
    const float radiusRatio = 0.38 + scenarioRandom01(key ^ 0x27d4eb2du) * 0.32;
    const float radius = worldRadius * radiusRatio;
    return vec2(cos(angle), sin(angle)) * radius;
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
