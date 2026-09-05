// The trail field: a decaying deposit agents and beacons leave on the ground,
// and the addressing both languages use to reach into it.
//
// Compiled twice, like ScenarioKernel.inl and BrainKernel.inl -- as C++ through
// TrailKernel.hpp and as GLSL through shaders/simulation/trail_kernel.glsl. Same
// common-subset rules: VKEXP_TRAIL_FN in front of every function, only uint/bool
// and vec2 across boundaries, `f` on every float literal.
//
// Why a storage buffer and not an image. Every other simulation structure here
// is an SSBO, so this needs no new descriptor type, layout transition or format.
// More importantly deposits are scattered writes, and integer atomicAdd is
// commutative and exact: the order the GPU happens to schedule agents in cannot
// change the resulting field. A float image blended non-atomically would make
// the field order-dependent, and runMultiAgentDeterminism would start flickering
// instead of catching real races.

// --- storage -----------------------------------------------------------------

const uint TrailChannels = 3u; // RGB

// Fixed point. 1.0 is stored as this, so a deposit is an integer add. The
// ceiling matters: the decay pass multiplies in float, which is exact for
// integers below 2^24, i.e. 256x full scale. A cell saturated by a dozen agents
// sits two orders of magnitude below that.
const float TrailFixedPointScale = 65536.0f;
const uint TrailCellCeiling = 1u << 28;

// The ground resolution itself is a runtime parameter reaching both languages
// through the step block, so it is not declared here. Its bounds are expressed in
// body diameters rather than metres and live with the body radius in
// AgentTypes.hpp: what decides whether a resolution is useful is how many cells
// wide a track comes out, and "a fifth of an agent" says that where "9 mm" does
// not.

VKEXP_TRAIL_FN uint trailValueIndex(uint world, uint cellsPerWorld, uint cell, uint channel) {
    return ((world * cellsPerWorld) + cell) * TrailChannels + channel;
}

// --- addressing --------------------------------------------------------------

// World-space position to a cell coordinate. The field covers the bounding
// square of the arena, so a circular world leaves the corners unused -- the same
// trade the neighbour grid already makes.
VKEXP_TRAIL_FN uint trailCellAxis(float coordinate, float worldRadius, float cellSize, uint width) {
    const float shifted = (coordinate + worldRadius) / cellSize;
    if (shifted <= 0.0f) {
        return 0u;
    }
    const uint index = uint(shifted);
    return index >= width ? width - 1u : index;
}

VKEXP_TRAIL_FN uint trailCellIndex(vec2 position, float worldRadius, float cellSize, uint width) {
    const uint x = trailCellAxis(position.x, worldRadius, cellSize, width);
    const uint y = trailCellAxis(position.y, worldRadius, cellSize, width);
    return y * width + x;
}

// True when a point is inside the field at all. Positions are clamped rather
// than dropped for reading, but a deposit outside the arena must not fold back
// onto the edge and create a scent wall.
VKEXP_TRAIL_FN bool trailContains(vec2 position, float worldRadius) {
    return position.x >= -worldRadius && position.x <= worldRadius && position.y >= -worldRadius &&
           position.y <= worldRadius;
}

// --- dynamics ----------------------------------------------------------------

// Per-step survival factor from a per-second decay rate. Written in the same
// exp(-rate * dt) form as the drags and the contact solver, so the step rate is
// not smuggled into how long a trail lasts.
VKEXP_TRAIL_FN float trailSurvival(float decayRate, float deltaTime) {
    return exp(-decayRate * deltaTime);
}

// A half-life reads better than a rate in the UI, and this is the only place
// that conversion lives.
VKEXP_TRAIL_FN float trailDecayRateForHalfLife(float halfLifeSeconds) {
    return halfLifeSeconds > 0.0f ? 0.6931472f / halfLifeSeconds : 0.0f;
}

// Faint mark every agent leaves, before its signal is added on top: a silent
// agent still leaves a track, which is what makes a path followable at all.
const float TrailAgentBaseIntensity = 0.25f;
