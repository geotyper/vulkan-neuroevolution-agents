#include "vkexp/worlds/scenarios/ChainScenario.hpp"

#include "vkexp/simulation/Units.hpp"

#include <algorithm>
#include <cmath>

// The chain world. There is nothing to find here: no beacon, no cargo, no door.
// The only thing in the arena is the other agents, and the only thing scored is
// how many of them you are next to.
//
// Two rules make that a chain rather than a heap.
//
// The first is the speed floor, which is physics rather than scenery and lives
// with the drags. Without it the best answer is to stop in a good arrangement
// and never move again, and the world becomes a one-off packing puzzle. With a
// floor no arrangement can be parked into -- it has to be flown, continuously,
// while everyone else is also moving -- so what is being evolved is a rule for
// staying in formation rather than a place to stand.
//
// The second is the shape of the reward. Scoring merely "has a neighbour" rates
// two agents stuck together as highly as a column of ten, and there is then
// nothing to gain by lining up; every pair is already at the optimum. So the
// reward rises to a band and stops: at a band of two, the inside of a chain
// scores above its ends, which is the same statement as "be in a line". Above
// the crowd limit it turns into a penalty, because three neighbours this close
// is already a triangle and a triangle is how a heap starts.
//
// Both numbers are sliders, because which of them produces a chain is exactly
// what this world is for finding out. A band of one is the flat rule, and it is
// worth running as the control: if it chains as well as a band of two, the band
// is not what is doing the work.
//
// -- Why this world is not in the parity test --
//
// The neighbour count comes from the grid sweep in agent_step.comp, and there is
// no such sweep on the CPU: CpuSimulation.cpp resolves walls and nothing else,
// so it cannot answer "who is near me" at all. The accumulation therefore lives
// only in chain.glsl and afterStep here is deliberately null -- not a stub that
// would compute zero and look like agreement. That is the same arrangement the
// gate world already has for its reduction over the grid, and it is why neither
// world appears in runTrajectoryParity. What covers this one instead is a test
// that stages a known arrangement of agents, steps once, and reads the counts
// back.
namespace vkexp::worlds::chain {
namespace {

// Half the nominal trial spent in the chain is what counts as having done the
// task. Half rather than all because the population starts scattered and the
// first seconds of every trial are spent finding each other, which is work the
// world asks for and should not be scored as failure.
constexpr float achievedFraction = 0.5F;
constexpr std::uint32_t nominalSteps = 900;

std::uint32_t achievedObjectives(const AgentState& agent);

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    // metrics.w is the accumulated score in point-seconds and penalties.y the
    // seconds spent in the chain; both are laid down by chain.glsl. See the note
    // at the top of this file for why they are written there and only there.
    const float held = agent.metrics.w;
    const float bonus = achievedObjectives(agent) != 0 ? weights.objectiveBonus : 0.0F;
    return held + bonus - agent.metrics.z * weights.motorCostWeight - agent.penalties.x;
}

std::uint32_t achievedObjectives(const AgentState& agent) {
    const float required =
        achievedFraction * static_cast<float>(nominalSteps) * units::fixedTimeStep;
    return agent.penalties.y >= required ? 1U : 0U;
}

// No beacons at all, which is the point: the receptors see other agents' light
// and nothing else, so every gradient in this world is another agent.
ActiveBeacons beacons(const AgentState&, const SimulationStep&) { return {}; }

// Supplied rather than left null, because the shared fallback means "distance to
// the nearest beacon" and there are none. Zero says the world has no target, and
// the progress shaping that reads it then has nothing to shape -- which is
// correct here, since arriving somewhere is not what is being asked.
float targetDistance(const AgentState&, const SimulationStep&) { return 0.0F; }

ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    ScenarioParameterBlock block{};
    block.floats0.x = settings.chainCrowdPenalty;
    block.integers[0] = settings.chainRewardBand;
    block.integers[1] = settings.chainCrowdLimit;
    return block;
}

// The full sensor suite and both memory cells. A formation has to be held over
// time and the agent has to know which way it was going, so trimming the
// recurrent block here would be taking away the thing the task is about.
constexpr neuro::BrainShape brain{neuro::Topology::inputCount, neuro::Topology::defaultHiddenCount,
                                  neuro::Topology::outputCount};
static_assert(brain.fitsCapacity());

} // namespace

float stepScore(const std::uint32_t neighbours, const std::uint32_t rewardBand,
                const std::uint32_t crowdLimit, const float crowdPenalty) {
    const auto reward = static_cast<float>(std::min(neighbours, rewardBand));
    const auto crowd =
        static_cast<float>(neighbours > crowdLimit ? neighbours - crowdLimit : 0U);
    return reward - crowdPenalty * crowd;
}

bool inBand(const std::uint32_t neighbours, const std::uint32_t crowdLimit) {
    return neighbours >= 1U && neighbours <= crowdLimit;
}

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Chain",
        .key = "chain",
        .id = BeaconScenario::Chain,
        .brain = brain,
        .tunables = {.chainNeighbours = true},
        .objectiveLabel = "Held the chain",
        .description = "No beacon and no rest: score comes from who is beside you, and a body "
                       "that cannot stop has to hold formation while moving",
        .beacons = beacons,
        .beaconCount = 0,
        .targetDistance = targetDistance,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = 1,
        .nominalStepsPerGeneration = nominalSteps,
        .beforeStep = nullptr,
        // Null on purpose. The count this world scores does not exist on the CPU
        // side, so the accumulation is in chain.glsl alone rather than written
        // twice with one copy quietly producing zero.
        .afterStep = nullptr,
        .gpuParameters = gpuParameters,
    };
    return value;
}

} // namespace vkexp::worlds::chain
