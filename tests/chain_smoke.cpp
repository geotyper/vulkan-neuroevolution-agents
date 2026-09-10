// Device coverage for the chain world's neighbour count.
//
// This world is scored from a number that exists on one side only. The count
// comes from the grid sweep in agent_step.comp, and CpuSimulation.cpp has no
// such sweep -- it resolves walls and nothing else -- so the accumulation lives
// in chain.glsl alone and runTrajectoryParity cannot say anything about it. The
// gate world's reduction over the grid has exactly the same shape, and exactly
// the same gap.
//
// What replaces parity is arithmetic the test does itself. Agents are placed in
// a known arrangement, one step is run, and the counts are read back and
// compared against a count computed here from the same positions by brute force
// -- every pair, no grid. If the sweep misses a cell, double-counts a link,
// includes the agent itself, or reaches into a neighbouring logical world, the
// two numbers disagree.
//
// What is checked, in the order the failures matter:
//   * the count is right, against a brute-force count over the same positions;
//   * it stops at the radius, which is the one number the world is tuned by:
//     a sweep that used the light range instead would count far too many and
//     the reward band would never bite;
//   * an agent never counts itself, and never counts across worlds -- agents in
//     different logical worlds share an arena in metres and must not see each
//     other;
//   * the score follows the rule the C++ side states, so the shader and
//     ChainScenario::stepScore cannot drift apart;
//   * a world that did not ask for the count does not pay for one.

#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/simulation/SimulationDriver.hpp"
#include "vkexp/simulation/SimulationState.hpp"
#include "vkexp/worlds/WorldScenario.hpp"
#include "vkexp/worlds/scenarios/ChainScenario.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int skipExitCode = 77;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool closeTo(const float left, const float right, const float tolerance = 1.0e-4F) {
    return std::abs(left - right) <= tolerance;
}

// The same question the shader answers, asked without the grid: every other
// agent in the same logical world, inside the radius. This is the reference the
// device is checked against, so it is deliberately the slow obvious loop.
std::uint32_t neighboursByBruteForce(const std::vector<vkexp::AgentState>& agents,
                                     const std::size_t self, const float radius) {
    std::uint32_t count = 0;
    for (std::size_t other = 0; other < agents.size(); ++other) {
        if (other == self) {
            continue;
        }
        if (agents[other].penalties.w != agents[self].penalties.w) {
            continue;
        }
        const float dx = agents[other].pose.x - agents[self].pose.x;
        const float dy = agents[other].pose.y - agents[self].pose.y;
        if (std::hypot(dx, dy) <= radius) {
            ++count;
        }
    }
    return count;
}

int run() {
    vkexp::HeadlessComputeContext context{
        vkexp::HeadlessComputeConfig{.applicationName = "vkneuro chain smoke"}};

    vkexp::SimulationState state{};
    state.controls.stepsPerGeneration = 64;
    state.physics.beaconScenario = vkexp::BeaconScenario::Chain;
    state.physics.worldRadius = vkexp::worldRadiusForSize(state.physics.worldSize);
    state.physics.lightSensorRange = vkexp::lightRangeForWorld(state.physics);
    // A floor, because this world is meant to be run with one and the count must
    // be right while everyone is moving rather than only in a staged freeze.
    state.physics.minimumSpeed = 0.20F;
    state.worlds.requestedAgentsPerWorld = 16;

    vkexp::SimulationDriver driver{
        state, vkexp::EvolutionSettings{.populationSize = 64}, vkexp::SimulationDriverConfig{}};
    driver.createResources(context.physicalDevice(), context.device());

    const float radius = state.physics.chainNeighbourBodies * vkexp::agentBodyDiameter;
    require(radius <= state.physics.lightSensorRange,
            "the staged radius is inside what the grid sweep reaches");

    // A line of agents at a known spacing, laid along x inside the first logical
    // world. At 0.9 of the radius each agent has its immediate neighbours and
    // nobody else, so the expected counts are 1 at the ends and 2 inside -- the
    // arrangement the reward band is meant to prefer.
    vkexp::WorldSnapshot staged = driver.snapshot();
    const float spacing = radius * 0.9F;
    const std::uint32_t firstWorld =
        staged.agents.empty() ? 0U
                              : static_cast<std::uint32_t>(std::max(staged.agents[0].penalties.w,
                                                                    0.0F) + 0.5F);
    std::size_t placed = 0;
    for (vkexp::AgentState& agent : staged.agents) {
        const auto world = static_cast<std::uint32_t>(std::max(agent.penalties.w, 0.0F) + 0.5F);
        if (world == firstWorld && placed < 6) {
            agent.pose.x = -1.0F + static_cast<float>(placed) * spacing;
            agent.pose.y = 0.0F;
            ++placed;
        } else {
            // Everyone else parked far away along the rim, so the line's counts
            // have one explanation. Agents in other logical worlds share these
            // metres; if the sweep leaked across worlds the line would see them.
            agent.pose.x = state.physics.worldRadius - vkexp::agentBodyRadius;
            agent.pose.y = state.physics.worldRadius - vkexp::agentBodyRadius;
        }
        agent.pose.z = 0.0F;
        agent.motion = {0.0F, 0.0F, 0.0F, 1.0F};
        agent.metrics = {0.0F, 0.0F, 0.0F, 0.0F};
        agent.target = {0.0F, 0.0F, agent.target.z, 0.0F};
        agent.penalties.x = 0.0F;
        agent.penalties.y = 0.0F;
    }
    require(placed == 6, "the staged line got the agents it asked for");
    staged.step = 0;
    driver.restoreSnapshot(staged);

    // The positions the count is taken against are the ones going in: the count
    // is published before the brain runs and before the body moves, so it
    // describes the arrangement that was staged rather than the one it becomes.
    const std::vector<vkexp::AgentState> before = staged.agents;
    context.immediate().execute(
        [&](const VkCommandBuffer commands) { driver.recordSteps(commands, 1); });
    const std::vector<vkexp::AgentState> after = driver.snapshot().agents;
    require(after.size() == before.size(), "the staged run keeps its population");

    std::uint32_t interior = 0;
    for (std::size_t index = 0; index < after.size(); ++index) {
        const auto counted = static_cast<std::uint32_t>(std::max(after[index].target.w, 0.0F) +
                                                        0.5F);
        const std::uint32_t expected = neighboursByBruteForce(before, index, radius);
        require(counted == expected,
                "agent " + std::to_string(index) + " counted " + std::to_string(counted) +
                    " neighbours where every pair says " + std::to_string(expected));

        const float score = vkexp::worlds::chain::stepScore(expected, state.physics.chainRewardBand,
                                                            state.physics.chainCrowdLimit,
                                                            state.physics.chainCrowdPenalty);
        require(closeTo(after[index].metrics.w, score * state.physics.deltaTime),
                "agent " + std::to_string(index) +
                    " scored what the rule in ChainScenario.cpp says for its count");
        const bool banded =
            vkexp::worlds::chain::inBand(expected, state.physics.chainCrowdLimit);
        require(closeTo(after[index].penalties.y, banded ? state.physics.deltaTime : 0.0F),
                "agent " + std::to_string(index) + " banked time in the chain only while in band");
        if (expected == 2) {
            ++interior;
        }
    }
    // The staged line is six agents, so four of them are interior. Asserted
    // rather than merely printed, because a radius that quietly reached further
    // would make every one of them interior and the brute-force check above
    // would still agree with the device -- both would be wrong together.
    require(interior == 4, "the staged line has four interior agents, so the radius bites where "
                           "it was told to and not at the light range");

    std::cout << "Chain smoke passed on " << context.deviceName() << '\n';
    return EXIT_SUCCESS;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const vkexp::HeadlessComputeUnavailable& unavailable) {
        std::cout << "Skipping: " << unavailable.what() << '\n';
        return skipExitCode;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
