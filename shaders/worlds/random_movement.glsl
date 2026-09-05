// floats0 = {wander speed, roam radius ratio, motion time, teleport probability},
// integers[0] = motion seed.
// Packed by gpuParameters in src/worlds/scenarios/RandomMovementScenario.cpp.
// Geometry from the shared kernel; see include/vkexp/worlds/ScenarioKernel.inl.
vec2 randomMovementScenarioPosition(uint trial, float worldRadius, ScenarioParameters sp) {
    const float teleportProbability = sp.floats0.w;
    const uint motionSeed = sp.integers.x;
    const float clampedTime = max(sp.floats0.z, 0.0);
    const uint segment = randomMotionSegment(clampedTime);
    uint epoch = 0u;
    for (uint candidate = segment; candidate > 0u; --candidate) {
        if (randomTeleportSegment(motionSeed, trial, candidate, teleportProbability)) {
            epoch = candidate;
            break;
        }
    }
    const float localTime = clampedTime - float(epoch) * RandomMotionSegmentSeconds;
    const float roamRadius = worldRadius * sp.floats0.y;
    const float scaledTime = localTime * sp.floats0.x / max(roamRadius, 0.001);
    const uint key = randomWanderKey(motionSeed, trial, epoch);
    return randomWanderOffset(key, scaledTime) * roamRadius;
}
