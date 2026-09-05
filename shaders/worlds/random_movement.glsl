// floats0 = {wander speed, roam radius ratio, motion time, teleport probability},
// integers[0] = motion seed.
// Packed by gpuParameters in src/worlds/scenarios/RandomMovementScenario.cpp.
bool randomMovementTeleportSegment(uint segment, uint trial, float teleportProbability,
                                   uint motionSeed) {
    if (segment == 0u || teleportProbability <= 0.0) {
        return false;
    }
    const uint eventKey = motionSeed ^ (trial * 0x27d4eb2du) ^
                          (segment * 0x165667b1u) ^ 0xa511e9b3u;
    return scenarioRandom01(eventKey) < teleportProbability;
}

vec2 randomMovementScenarioPosition(uint trial, float worldRadius, ScenarioParameters sp) {
    const float teleportProbability = sp.floats0.w;
    const uint motionSeed = sp.integers.x;
    const float clampedTime = max(sp.floats0.z, 0.0);
    const uint segment = uint(floor(clampedTime / 3.0));
    uint epoch = 0u;
    for (uint candidate = segment; candidate > 0u; --candidate) {
        if (randomMovementTeleportSegment(candidate, trial, teleportProbability, motionSeed)) {
            epoch = candidate;
            break;
        }
    }
    const float localTime = clampedTime - float(epoch) * 3.0;
    const float roamRadius = worldRadius * sp.floats0.y;
    const float scaledTime = localTime * sp.floats0.x / max(roamRadius, 0.001);
    const uint key = motionSeed ^ (trial * 0x9e3779b9u) ^ (epoch * 0x85ebca6bu);
    const float phase0 = scenarioRandom01(key) * Tau;
    const float phase1 = scenarioRandom01(key ^ 0x68bc21ebu) * Tau;
    const float phase2 = scenarioRandom01(key ^ 0x02e5be93u) * Tau;
    const float phase3 = scenarioRandom01(key ^ 0x967a889bu) * Tau;
    const float rawX = 0.62 * sin(scaledTime * 0.73 + phase0) +
                       0.28 * sin(scaledTime * 1.37 + phase1) +
                       0.18 * sin(scaledTime * 0.31 + phase2);
    const float rawY = 0.58 * sin(scaledTime * 0.83 + phase3) +
                       0.31 * sin(scaledTime * 1.19 + phase0) +
                       0.16 * sin(scaledTime * 0.27 + phase1);
    const vec2 rawPosition = vec2(rawX, rawY);
    return rawPosition * (roamRadius / (1.25 + length(rawPosition)));
}
