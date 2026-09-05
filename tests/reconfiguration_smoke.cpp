// Reconfiguration coverage for SimulationDriver.
//
// Changing the arena size, the group size or the trail resolution all resize GPU
// resources under a running simulation, and each of those hung the application at
// least once. The failures were not in the settings but in ownership: buffers
// sized for a worst case that no longer matched what was running, descriptors
// left pointing at a reallocated handle, and a coarsening loop that had to
// terminate. None of that is reachable from a single-configuration test, because
// a fresh driver always agrees with itself.
//
// So this walks the reconfigurations the UI can produce and steps the simulation
// after each one. A hang shows up as the ctest timeout rather than as a wedged
// desktop, and anything that survives the step is checked for having actually
// simulated rather than quietly produced NaNs.

#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/simulation/SimulationDriver.hpp"
#include "vkexp/simulation/SimulationState.hpp"

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

// Runs a short generation and checks the population actually moved through it.
void stepAndCheck(vkexp::HeadlessComputeContext& context, vkexp::SimulationDriver& driver,
                  vkexp::SimulationState& state, const std::string& what) {
    std::cout << "  " << what << ": " << state.worlds.worldCount << " worlds, trail "
              << state.trail.width << "x" << state.trail.width << " cells of "
              << state.physics.trailCellSize << " m" << std::endl;
    require(state.trail.buffer != VK_NULL_HANDLE, what + ": trail field is published");
    require(state.trail.cellsPerWorld > 0, what + ": trail field has cells");
    require(state.worlds.worldCount > 0, what + ": at least one logical world");

    while (!driver.generationComplete()) {
        context.immediate().execute(
            [&](const VkCommandBuffer commands) { driver.recordSteps(commands, 64); });
    }
    const vkexp::GenerationSummary summary = driver.finishGeneration();
    require(std::isfinite(summary.bestFitness) && std::isfinite(summary.medianFitness),
            what + ": generation produced finite fitness");

    // A field sized for a different configuration would leave the shader reading
    // or writing outside the agents' own world; the cheapest thing that catches
    // it is that every agent is still inside the arena it was given.
    for (const vkexp::AgentState& agent : driver.agents()) {
        require(std::isfinite(agent.pose.x) && std::isfinite(agent.pose.y),
                what + ": agent pose stayed finite");
        const float distance = std::hypot(agent.pose.x, agent.pose.y);
        require(distance <= state.physics.worldRadius * 1.5F + 1.0F,
                what + ": agent stayed inside its arena");
    }
}

int run() {
    vkexp::HeadlessComputeContext context{
        vkexp::HeadlessComputeConfig{.applicationName = "vkneuro reconfiguration smoke"}};

    vkexp::SimulationState state{};
    state.controls.stepsPerGeneration = 96;
    state.worlds.requestedAgentsPerWorld = 24;
    state.physics.worldSize = vkexp::WorldSize::Small;
    state.physics.worldRadius = vkexp::worldRadiusForSize(state.physics.worldSize);
    state.physics.lightSensorRange = vkexp::lightRangeForWorld(state.physics);

    vkexp::SimulationDriver driver{state, vkexp::EvolutionSettings{.populationSize = 96}, {}};
    driver.createResources(context.physicalDevice(), context.device());
    stepAndCheck(context, driver, state, "initial configuration");

    // 1. Arena size. The neighbour grid, the trail field and the light range are
    //    all derived from it, and all three used to be sized from a different
    //    assumption about which world was running.
    for (const vkexp::WorldSize size :
         {vkexp::WorldSize::Large, vkexp::WorldSize::Medium, vkexp::WorldSize::Small}) {
        state.physics.worldSize = size;
        state.physics.worldRadius = vkexp::worldRadiusForSize(size);
        state.physics.lightSensorRange = vkexp::lightRangeForWorld(state.physics);
        driver.restart();
        stepAndCheck(context, driver, state,
                     "world size " + std::to_string(static_cast<int>(size)));
    }

    // 2. Group size, in both directions. Shrinking it multiplies the number of
    //    logical worlds, which is what grows the per-world buffers; growing it
    //    back must not leave the driver believing in the larger layout.
    for (const std::uint32_t agentsPerWorld : {96U, 10U, 32U, 10U}) {
        state.worlds.requestedAgentsPerWorld = agentsPerWorld;
        driver.restart();
        require(state.worlds.agentsPerWorld <= agentsPerWorld,
                "group size " + std::to_string(agentsPerWorld) + ": clamped, never inflated");
        stepAndCheck(context, driver, state, "group size " + std::to_string(agentsPerWorld));
    }

    // 3. Trail resolution, finest first so the budget clamp is exercised while the
    //    world count is at its largest. The driver may coarsen the request; what
    //    it must not do is fail to terminate or keep a stale field.
    for (const float fraction :
         {vkexp::trailCellFractionFinest, 0.25F, 0.5F, vkexp::trailCellFractionCoarsest}) {
        state.physics.trailCellSize = vkexp::trailCellSizeForBodyFraction(fraction);
        driver.restart();
        require(state.physics.trailCellSize >=
                    vkexp::trailCellSizeForBodyFraction(vkexp::trailCellFractionFinest),
                "trail resolution: never refined past the finest setting");
        require(static_cast<std::uint64_t>(state.trail.cellsPerWorld) * state.worlds.worldCount *
                        vkexp::trail::kernel::TrailChannels * sizeof(std::uint32_t) <=
                    vkexp::trailFieldByteBudget,
                "trail resolution: field stays inside the byte budget");
        stepAndCheck(context, driver, state, "trail resolution");
    }

    // 4. The combination that changes both the world count and the field size at
    //    once, which is the shape of the reported failure.
    state.physics.worldSize = vkexp::WorldSize::Large;
    state.physics.worldRadius = vkexp::worldRadiusForSize(vkexp::WorldSize::Large);
    state.physics.lightSensorRange = vkexp::lightRangeForWorld(state.physics);
    state.worlds.requestedAgentsPerWorld = 10;
    state.physics.trailCellSize =
        vkexp::trailCellSizeForBodyFraction(vkexp::trailCellFractionFinest);
    driver.restart();
    stepAndCheck(context, driver, state, "large arena, smallest groups, finest trail");

    driver.destroyResources();
    std::cout << "Reconfiguration smoke passed on " << context.deviceName() << '\n';
    return EXIT_SUCCESS;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const vkexp::HeadlessComputeUnavailable& unavailable) {
        std::cout << "Skipping reconfiguration smoke: " << unavailable.what() << '\n';
        return skipExitCode;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
