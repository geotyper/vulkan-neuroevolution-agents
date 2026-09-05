// Scenario math shared verbatim by the CPU reference and the shaders.
//
// This file is compiled twice: once as C++ through ScenarioKernel.hpp, which
// supplies a small vec2/uint shim, and once as GLSL through
// shaders/worlds/scenario_kernel.glsl, where the same names are built in. It is
// therefore restricted to the common subset of both languages:
//
//   * every function is introduced with VKEXP_KERNEL_FN;
//   * every float literal carries the `f` suffix, so C++ does not silently
//     promote the arithmetic to double and drift away from the GPU;
//   * only vec2, float, uint and bool cross function boundaries;
//   * no swizzles, matrices, references or standard library calls.
//
// Anything that needs SimulationStep, AgentState or ScenarioParameters stays in
// the per-language scenario files and calls into here.

const float ScenarioTau = 6.28318530718f;
const float ForageHomeRelocationSeconds = 8.0f;
const float ForageHomeMinimumRadiusRatio = 0.38f;
const float ForageHomeRadiusRange = 0.32f;
const float RandomMotionSegmentSeconds = 3.0f;
const float AlternatingDiagonalRatio = 0.62f;

VKEXP_KERNEL_FN uint scenarioHash(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

VKEXP_KERNEL_FN float scenarioRandom01(uint value) {
    return float(scenarioHash(value) & 0x00ffffffu) / 16777215.0f;
}

// --- alternating diagonals -------------------------------------------------

// Unit-square offset of one diagonal beacon; callers scale by the world radius.
VKEXP_KERNEL_FN vec2 alternatingDiagonalOffset(uint beaconIndex, uint phase) {
    const float x = beaconIndex == 0u ? -AlternatingDiagonalRatio : AlternatingDiagonalRatio;
    const float first = phase == 0u ? -AlternatingDiagonalRatio : AlternatingDiagonalRatio;
    const float y = beaconIndex == 0u ? first : -first;
    return vec2(x, y);
}

// --- rotating orbit --------------------------------------------------------

// Rotates a unit direction by `angle`. Callers pass the normalised base heading
// and scale the result by the orbit radius.
VKEXP_KERNEL_FN vec2 rotatingOrbitOffset(vec2 baseDirection, float angle) {
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return vec2(baseDirection.x * cosine - baseDirection.y * sine,
                baseDirection.x * sine + baseDirection.y * cosine);
}

// --- forage home -----------------------------------------------------------

VKEXP_KERNEL_FN uint forageHomeEpoch(float motionTime) {
    return uint(floor(max(motionTime, 0.0f) / ForageHomeRelocationSeconds));
}

VKEXP_KERNEL_FN bool forageHomeRelocated(float motionTime, float deltaTime) {
    const float currentTime = max(motionTime, 0.0f);
    const float previousTime = max(currentTime - deltaTime, 0.0f);
    const float epsilon = max(deltaTime * 0.01f, 0.000001f);
    return uint(floor((currentTime + epsilon) / ForageHomeRelocationSeconds)) !=
           uint(floor((previousTime + epsilon) / ForageHomeRelocationSeconds));
}

VKEXP_KERNEL_FN uint forageHomeKey(uint motionSeed, uint trial, uint epoch) {
    return motionSeed ^ (trial * 0x51ed270bu) ^ (epoch * 0x85ebca6bu) ^ 0xc2b2ae35u;
}

// Home position as a fraction of the world radius.
VKEXP_KERNEL_FN vec2 forageHomeOffset(uint key) {
    const float angle = scenarioRandom01(key) * ScenarioTau;
    const float radiusRatio =
        ForageHomeMinimumRadiusRatio + scenarioRandom01(key ^ 0x27d4eb2du) * ForageHomeRadiusRange;
    return vec2(cos(angle) * radiusRatio, sin(angle) * radiusRatio);
}

// --- random movement -------------------------------------------------------

VKEXP_KERNEL_FN uint randomMotionSegment(float motionTime) {
    return uint(floor(max(motionTime, 0.0f) / RandomMotionSegmentSeconds));
}

VKEXP_KERNEL_FN uint randomTeleportKey(uint motionSeed, uint trial, uint segment) {
    return motionSeed ^ (trial * 0x27d4eb2du) ^ (segment * 0x165667b1u) ^ 0xa511e9b3u;
}

VKEXP_KERNEL_FN bool randomTeleportSegment(uint motionSeed, uint trial, uint segment,
                                           float teleportProbability) {
    if (segment == 0u || teleportProbability <= 0.0f) {
        return false;
    }
    return scenarioRandom01(randomTeleportKey(motionSeed, trial, segment)) < teleportProbability;
}

VKEXP_KERNEL_FN uint randomWanderKey(uint motionSeed, uint trial, uint epoch) {
    return motionSeed ^ (trial * 0x9e3779b9u) ^ (epoch * 0x85ebca6bu);
}

// Wander position as a fraction of the roam radius.
VKEXP_KERNEL_FN vec2 randomWanderOffset(uint key, float scaledTime) {
    const float phase0 = scenarioRandom01(key) * ScenarioTau;
    const float phase1 = scenarioRandom01(key ^ 0x68bc21ebu) * ScenarioTau;
    const float phase2 = scenarioRandom01(key ^ 0x02e5be93u) * ScenarioTau;
    const float phase3 = scenarioRandom01(key ^ 0x967a889bu) * ScenarioTau;
    const float rawX = 0.62f * sin(scaledTime * 0.73f + phase0) +
                       0.28f * sin(scaledTime * 1.37f + phase1) +
                       0.18f * sin(scaledTime * 0.31f + phase2);
    const float rawY = 0.58f * sin(scaledTime * 0.83f + phase3) +
                       0.31f * sin(scaledTime * 1.19f + phase0) +
                       0.16f * sin(scaledTime * 0.27f + phase1);
    const vec2 raw = vec2(rawX, rawY);
    const float scale = 1.0f / (1.25f + length(raw));
    return vec2(raw.x * scale, raw.y * scale);
}
