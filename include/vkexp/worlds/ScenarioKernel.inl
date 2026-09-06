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
const float BeaconVisualRadius = 0.060f;

// --- two doors -------------------------------------------------------------
//
// A wall across the arena with two gaps. One leads through to the resource; the
// other opens into a closed pocket. From the home side the two are identical, so
// the only way to know is to have gone -- which is what makes the world worth
// building: a memory has something to hold, and a mark has something to say.
//
// The geometry is derived from the arena radius rather than stored, so it costs
// no parameter slot and the CPU and the shader cannot disagree about where a
// wall is. Every box is axis-aligned, which keeps the contact test to a clamp
// and a subtraction in both languages.
//
// Layout, in fractions of the world radius:
//
//        resource at (0, +0.72)
//     +-------------------------------+
//     |          #####                |   <- pocket cap over the blocked door
//     |          #   #                |   <- pocket sides
//     |######  ##########  ###########|   <- the wall, at y = 0
//     |      A            B           |
//     |          home at (0, -0.72)   |
//     +-------------------------------+

const uint TwoDoorsBoxCount = 6u;
const float TwoDoorsWallHalfThickness = 0.045f;
const float TwoDoorsDoorOffset = 0.40f;
const float TwoDoorsDoorHalfWidth = 0.07f;
const float TwoDoorsPocketDepth = 0.30f;
const float TwoDoorsArenaReach = 1.05f; // past the arena edge, so no gap at the rim
const float TwoDoorsHomeY = -0.72f;
const float TwoDoorsResourceY = 0.72f;

// Which gap is a dead end this trial. Alternating by trial means a genome is
// scored on both, so it cannot win by always turning the same way -- and the
// trial index is not among the network's inputs, so it cannot be read off.
VKEXP_KERNEL_FN uint twoDoorsBlockedDoor(uint trial) { return trial & 1u; }

// Signed x of a door centre: door 0 left, door 1 right.
VKEXP_KERNEL_FN float twoDoorsDoorCentre(uint door, float worldRadius) {
    const float side = door == 0u ? -TwoDoorsDoorOffset : TwoDoorsDoorOffset;
    return side * worldRadius;
}

VKEXP_KERNEL_FN vec2 twoDoorsHomePosition(float worldRadius) {
    return vec2(0.0f, TwoDoorsHomeY * worldRadius);
}

VKEXP_KERNEL_FN vec2 twoDoorsResourcePosition(float worldRadius) {
    return vec2(0.0f, TwoDoorsResourceY * worldRadius);
}

// Box `index` as a centre; `twoDoorsBoxHalfExtent` gives the matching extent.
// Two calls rather than one because only vec2, float, uint and bool may cross a
// shared-kernel boundary, and a box is four numbers.
VKEXP_KERNEL_FN vec2 twoDoorsBoxCentre(uint index, float worldRadius, uint blockedDoor) {
    const float leftDoor = twoDoorsDoorCentre(0u, worldRadius);
    const float rightDoor = twoDoorsDoorCentre(1u, worldRadius);
    const float doorHalf = TwoDoorsDoorHalfWidth * worldRadius;
    const float reach = TwoDoorsArenaReach * worldRadius;
    const float side = TwoDoorsWallHalfThickness * worldRadius;
    const float blockedX = twoDoorsDoorCentre(blockedDoor, worldRadius);
    if (index == 0u) { // wall, outer left
        return vec2((-reach + (leftDoor - doorHalf)) * 0.5f, 0.0f);
    }
    if (index == 1u) { // wall, between the doors
        return vec2(0.0f, 0.0f);
    }
    if (index == 2u) { // wall, outer right
        return vec2(((rightDoor + doorHalf) + reach) * 0.5f, 0.0f);
    }
    if (index == 3u) { // pocket cap
        return vec2(blockedX, TwoDoorsPocketDepth * worldRadius);
    }
    if (index == 4u) { // pocket side, left of the blocked door
        return vec2(blockedX - doorHalf - side, TwoDoorsPocketDepth * worldRadius * 0.5f);
    }
    return vec2(blockedX + doorHalf + side, TwoDoorsPocketDepth * worldRadius * 0.5f);
}

// No blockedDoor here: which gap is the dead end moves the pocket, never its
// size, and an unused parameter would only invite one to be passed wrongly.
VKEXP_KERNEL_FN vec2 twoDoorsBoxHalfExtent(uint index, float worldRadius) {
    const float leftDoor = twoDoorsDoorCentre(0u, worldRadius);
    const float rightDoor = twoDoorsDoorCentre(1u, worldRadius);
    const float doorHalf = TwoDoorsDoorHalfWidth * worldRadius;
    const float reach = TwoDoorsArenaReach * worldRadius;
    const float thickness = TwoDoorsWallHalfThickness * worldRadius;
    if (index == 0u) {
        return vec2(((leftDoor - doorHalf) + reach) * 0.5f, thickness);
    }
    if (index == 1u) {
        // Centred on the origin by construction, so its half width is just the
        // inner edge of the right-hand door.
        return vec2(rightDoor - doorHalf, thickness);
    }
    if (index == 2u) {
        return vec2((reach - (rightDoor + doorHalf)) * 0.5f, thickness);
    }
    if (index == 3u) {
        return vec2(doorHalf + 2.0f * thickness, thickness);
    }
    // The sides run from the wall up to the cap, so the pocket is closed on
    // three sides and the only way out is back through the door.
    return vec2(thickness, TwoDoorsPocketDepth * worldRadius * 0.5f);
}

// Does the segment from `start` to `finish` cross the box? The slab test, which
// is the whole of light occlusion: a wall that stops a body but not its light is
// a wall an agent can see through, and the light gradient then pulls it straight
// into the one place it cannot go.
//
// A lightmap would answer the same question by sampling, and would be the right
// tool for hundreds of sources over complex geometry. Here there are at most two
// beacons and six boxes, and the receptors are directional -- a map gives the
// light at a point and loses the direction the sharp receptor tuning needs, so
// it would have to be marched along each ray anyway. Twelve slab tests per agent
// per step sit beside a brain that already does more than a thousand multiplies.
VKEXP_KERNEL_FN bool segmentHitsBox(vec2 start, vec2 finish, vec2 centre, vec2 halfExtent) {
    float enter = 0.0f;
    float leave = 1.0f;

    const float spanX = finish.x - start.x;
    const float lowX = centre.x - halfExtent.x;
    const float highX = centre.x + halfExtent.x;
    if (spanX > -1.0e-8f && spanX < 1.0e-8f) {
        // Parallel to the slab: either the whole segment is inside it or the box
        // cannot be crossed at all.
        if (start.x < lowX || start.x > highX) {
            return false;
        }
    } else {
        float nearX = (lowX - start.x) / spanX;
        float farX = (highX - start.x) / spanX;
        if (nearX > farX) {
            const float held = nearX;
            nearX = farX;
            farX = held;
        }
        enter = max(enter, nearX);
        leave = leave < farX ? leave : farX;
    }

    const float spanY = finish.y - start.y;
    const float lowY = centre.y - halfExtent.y;
    const float highY = centre.y + halfExtent.y;
    if (spanY > -1.0e-8f && spanY < 1.0e-8f) {
        if (start.y < lowY || start.y > highY) {
            return false;
        }
    } else {
        float nearY = (lowY - start.y) / spanY;
        float farY = (highY - start.y) / spanY;
        if (nearY > farY) {
            const float held = nearY;
            nearY = farY;
            farY = held;
        }
        enter = max(enter, nearY);
        leave = leave < farY ? leave : farY;
    }

    return enter <= leave;
}

// Push-out for a circle against an axis-aligned box: zero when clear, otherwise
// the shortest vector that separates them. Written as the shallowest overlap
// axis rather than as a nearest-point normal so an agent that has sunk into a
// wall leaves the way it came instead of being ejected through it.
VKEXP_KERNEL_FN vec2 boxPushOut(vec2 position, float radius, vec2 centre, vec2 halfExtent) {
    const float dx = position.x - centre.x;
    const float dy = position.y - centre.y;
    const float overlapX = halfExtent.x + radius - (dx < 0.0f ? -dx : dx);
    const float overlapY = halfExtent.y + radius - (dy < 0.0f ? -dy : dy);
    if (overlapX <= 0.0f || overlapY <= 0.0f) {
        return vec2(0.0f, 0.0f);
    }
    if (overlapX < overlapY) {
        return vec2(dx < 0.0f ? -overlapX : overlapX, 0.0f);
    }
    return vec2(0.0f, dy < 0.0f ? -overlapY : overlapY);
}

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
