#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/profiling/CpuProfiler.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/CpuSimulation.hpp"
#include "vkexp/simulation/Sensors.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

bool closeTo(const float left, const float right) { return std::abs(left - right) < 0.0001F; }

void testTimingSeries() {
    vkexp::TimingSeries series;
    series.add(1.0);
    series.add(2.0);
    series.add(3.0);
    series.add(4.0);

    const auto statistics = series.statistics();
    check(statistics.sampleCount == 4, "TimingSeries sample count");
    check(closeTo(statistics.currentMs, 4.0F), "TimingSeries current value");
    check(closeTo(statistics.averageMs, 2.5F), "TimingSeries average");
    check(closeTo(statistics.minimumMs, 1.0F), "TimingSeries minimum");
    check(closeTo(statistics.maximumMs, 4.0F), "TimingSeries maximum");
    check(closeTo(statistics.percentile95Ms, 4.0F), "TimingSeries p95");
}

void testCpuProfiler() {
    constexpr vkexp::ProfileMetricId frameMetric = 0;
    constexpr vkexp::ProfileMetricId cpuWorkMetric = 1;
    constexpr vkexp::ProfileMetricId customMetric = 2;
    vkexp::CpuProfiler profiler;
    profiler.beginFrame();
    profiler.addDuration(customMetric, 1.25);
    const vkexp::CpuProfiler::FrameSample sample = profiler.endFrame(frameMetric, cpuWorkMetric, 3);

    check(sample.wallMilliseconds >= 0.0, "CPU profiler wall time");
    check(sample.cpuMilliseconds >= 0.0, "CPU profiler process time");
    check(profiler.series(frameMetric).statistics().sampleCount == 1, "CPU frame sample");
    check(profiler.series(cpuWorkMetric).statistics().sampleCount == 1, "CPU work sample");
    check(closeTo(profiler.series(customMetric).statistics().currentMs, 1.25F),
          "CPU custom duration");
}

void testDispatchSize() {
    check(vkexp::divideRoundUp(17, 8) == 3, "Rounded-up integer division");
    check(vkexp::divideRoundUp(16, 8) == 2, "Exact integer division");

    const vkexp::DispatchSize groups = vkexp::dispatchSize({1921, 1081, 1}, {8, 8, 1});
    check(groups.x == 241, "Dispatch width");
    check(groups.y == 136, "Dispatch height");
    check(groups.z == 1, "Dispatch depth");

    bool rejectedZero = false;
    try {
        static_cast<void>(vkexp::dispatchSize({1, 1, 1}, {0, 1, 1}));
    } catch (const std::exception&) {
        rejectedZero = true;
    }
    check(rejectedZero, "Zero local size rejection");

    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupCount[0] = 1024;
    limits.maxComputeWorkGroupCount[1] = 1024;
    limits.maxComputeWorkGroupCount[2] = 64;
    limits.maxComputeWorkGroupSize[0] = 1024;
    limits.maxComputeWorkGroupSize[1] = 1024;
    limits.maxComputeWorkGroupSize[2] = 64;
    limits.maxComputeWorkGroupInvocations = 1024;
    limits.maxPushConstantsSize = 128;
    limits.maxStorageBufferRange = 4096;
    const std::array<VkDeviceSize, 2> validRanges{1024, 2048};
    vkexp::validateComputeLimits(limits, groups, {8, 8, 1}, 16, validRanges);

    bool rejectedGroupCount = false;
    try {
        vkexp::validateComputeLimits(limits, {1025, 1, 1}, {8, 8, 1});
    } catch (const std::exception&) {
        rejectedGroupCount = true;
    }
    check(rejectedGroupCount, "Dispatch group limit rejection");

    bool rejectedInvocations = false;
    try {
        vkexp::validateComputeLimits(limits, {1, 1, 1}, {64, 64, 1});
    } catch (const std::exception&) {
        rejectedInvocations = true;
    }
    check(rejectedInvocations, "Local invocation limit rejection");

    bool rejectedPushConstants = false;
    try {
        vkexp::validateComputeLimits(limits, {1, 1, 1}, {8, 8, 1}, 132);
    } catch (const std::exception&) {
        rejectedPushConstants = true;
    }
    check(rejectedPushConstants, "Push constant limit rejection");

    const std::array<VkDeviceSize, 1> oversizedRange{8192};
    bool rejectedStorageRange = false;
    try {
        vkexp::validateComputeLimits(limits, {1, 1, 1}, {8, 8, 1}, 0, oversizedRange);
    } catch (const std::exception&) {
        rejectedStorageRange = true;
    }
    check(rejectedStorageRange, "Storage buffer range limit rejection");
}

void testComputeResourceValidation() {
    check(vkexp::tightlyPackedImageSize(VK_FORMAT_R8G8B8A8_UNORM, {4, 4}) == 64,
          "RGBA8 tightly-packed image size");
    check(vkexp::tightlyPackedImageSize(VK_FORMAT_R32G32_SFLOAT, {3, 2}) == 48,
          "RG32 tightly-packed image size");

    bool rejectedUnsupportedFormat = false;
    try {
        static_cast<void>(vkexp::tightlyPackedImageSize(VK_FORMAT_D32_SFLOAT, {4, 4}));
    } catch (const std::exception&) {
        rejectedUnsupportedFormat = true;
    }
    check(rejectedUnsupportedFormat, "Unsupported image transfer format rejection");

    vkexp::ComputePipelineBuilder builder{VK_NULL_HANDLE, VK_NULL_HANDLE};
    builder.specializationConstant(7, std::uint32_t{42});
    bool rejectedDuplicateConstant = false;
    try {
        builder.specializationConstant(7, std::uint32_t{43});
    } catch (const std::exception&) {
        rejectedDuplicateConstant = true;
    }
    check(rejectedDuplicateConstant, "Duplicate specialization constant rejection");

    bool rejectedUnavailableDescriptorSet = false;
    try {
        static_cast<void>(vkexp::PingPongDescriptorSets{}.forReadIndex(0));
    } catch (const std::exception&) {
        rejectedUnavailableDescriptorSet = true;
    }
    check(rejectedUnavailableDescriptorSet, "Unavailable ping-pong descriptor rejection");
}

void testLogicalWorldPartition() {
    constexpr std::uint32_t genomes = 25;
    constexpr std::uint32_t agentsPerWorld = 10;
    constexpr std::uint32_t trials = 4;
    check(vkexp::clampAgentsPerWorld(genomes, 0) == vkexp::minimumAgentsPerWorld,
          "World partition clamps an empty group size");
    check(vkexp::clampAgentsPerWorld(genomes, 100) == genomes,
          "World partition supports all agents in one group");
    check(vkexp::worldGroupCount(genomes, agentsPerWorld) == 3,
          "World partition rounds up the group count");
    check(vkexp::logicalWorldCount(genomes, agentsPerWorld, trials) == 12,
          "World partition creates one world per group and trial");
    check(vkexp::logicalWorldForAgent(39, agentsPerWorld, trials) == 3,
          "Last agent in the first group stays in its trial world");
    check(vkexp::logicalWorldForAgent(40, agentsPerWorld, trials) == 4,
          "First agent in the second group enters the next set of worlds");
    check(vkexp::logicalWorldForAgent(98, agentsPerWorld, trials) == 10,
          "Partial final group maps to the expected trial world");
    check(vkexp::agentsInLogicalWorld(genomes, agentsPerWorld, trials, 0) == 10,
          "Full logical world reports its agent count");
    check(vkexp::agentsInLogicalWorld(genomes, agentsPerWorld, trials, 8) == 5,
          "Partial logical world reports its agent count");
    check(vkexp::logicalWorldCount(genomes, genomes, trials) == trials,
          "All-agent mode preserves only the evaluation trial worlds");
}

void testPingPongState() {
    vkexp::PingPongBuffer buffers;
    check(buffers.readIndex() == 0 && buffers.writeIndex() == 1, "Initial ping-pong indices");
    buffers.swap();
    check(buffers.readIndex() == 1 && buffers.writeIndex() == 0, "Swapped ping-pong indices");
    buffers.swap();
    check(buffers.readIndex() == 0 && buffers.writeIndex() == 1, "Restored ping-pong indices");
}

void testNeuralNetworkContract() {
    vkexp::neuro::Weights weights{};
    vkexp::neuro::Inputs inputs{};
    inputs.fill(1.0F);
    const vkexp::neuro::Outputs outputs = vkexp::neuro::evaluate(weights, inputs);
    for (const float output : outputs) {
        check(closeTo(output, 0.0F), "Zero neural network output");
    }
    check(vkexp::neuro::Topology::inputCount == 52, "Multimodal neural input count");
    check(vkexp::neuro::Topology::outputCount == 8, "Actuator and memory output count");
    check(vkexp::neuro::Topology::weightCount == 1228, "Stable flattened neural weight count");
}

void testMultimodalSensors() {
    vkexp::AgentState agent{};
    agent.pose = {0.0F, 0.0F, 0.0F, 0.022F};
    agent.motion.w = 1.0F;
    agent.target = {1.0F, 0.0F, 0.0F, 0.0F};
    agent.internal = {0.7F, 1.0F, -0.4F, 0.6F};
    agent.wallTouch0.x = 0.7F;
    agent.agentTouch1.w = 0.8F;
    const vkexp::SimulationStep settings{};
    const vkexp::neuro::Inputs inputs = vkexp::sampleAgentInputs(agent, settings);
    constexpr std::size_t center = 3 * vkexp::neuro::Topology::lightChannelsPerReceptor;
    check(inputs[center] > 0.0F && inputs[center + 1] > inputs[center],
          "RGB photoreceptor observes cyan beacon");
    check(inputs[center + 3] > 0.0F, "Photoreceptor luminance channel");
    constexpr std::size_t tactile = vkexp::neuro::Topology::lightReceptorCount *
                                    vkexp::neuro::Topology::lightChannelsPerReceptor;
    check(closeTo(inputs[tactile], 0.7F), "Wall tactile sector mapping");
    check(closeTo(inputs[tactile + 7 * 2 + 1], 0.8F), "Agent tactile sector mapping");
    constexpr std::size_t task = tactile + vkexp::neuro::Topology::tactileSectorCount *
                                               vkexp::neuro::Topology::tactileChannelsPerSector +
                                 vkexp::neuro::Topology::selfInputCount;
    check(closeTo(inputs[task], 0.7F) && closeTo(inputs[task + 1], 1.0F),
          "Cargo and task-state input mapping");
    check(closeTo(inputs[task + 2], -0.4F) && closeTo(inputs[task + 3], 0.6F),
          "Recurrent memory input mapping");
}

void testWorldAndBeaconScenarios() {
    check(closeTo(vkexp::worldRadiusForSize(vkexp::WorldSize::Small), 1.84F), "Small world radius");
    check(closeTo(vkexp::worldRadiusForSize(vkexp::WorldSize::Medium), 1.84F * 1.5F),
          "Medium world radius");
    check(closeTo(vkexp::worldRadiusForSize(vkexp::WorldSize::Large), 1.84F * 3.0F),
          "Large world radius");
    vkexp::SimulationStep arrivalSettings{};
    arrivalSettings.arrivalRadiusMultiplier = 0.1F;
    check(closeTo(vkexp::beaconArrivalRadius(arrivalSettings),
                  vkexp::beaconVisualRadius * 0.1F),
          "Minimum arrival multiplier scales the beacon radius");
    arrivalSettings.arrivalRadiusMultiplier = 5.0F;
    check(closeTo(vkexp::beaconArrivalRadius(arrivalSettings),
                  vkexp::beaconVisualRadius * 5.0F),
          "Maximum arrival multiplier scales the beacon radius");

    vkexp::AgentState agent{};
    agent.target = {1.0F, 0.5F, 2.0F, 0.0F};
    vkexp::SimulationStep settings{};
    const vkexp::ActiveBeacons stationary = vkexp::activeBeacons(agent, settings);
    check(stationary.count == 1 && closeTo(stationary.values[0].position.x, 1.0F) &&
              closeTo(stationary.values[0].position.y, 0.5F),
          "Stationary beacon uses the trial target");

    settings.beaconScenario = vkexp::BeaconScenario::AlternatingDiagonals;
    settings.beaconPhase = 0;
    const vkexp::ActiveBeacons firstDiagonal = vkexp::activeBeacons(agent, settings);
    settings.beaconPhase = 1;
    const vkexp::ActiveBeacons secondDiagonal = vkexp::activeBeacons(agent, settings);
    check(firstDiagonal.count == 2 &&
              firstDiagonal.values[0].position.x * firstDiagonal.values[0].position.y > 0.0F &&
              firstDiagonal.values[1].position.x * firstDiagonal.values[1].position.y > 0.0F,
          "First beacon pair occupies one diagonal");
    check(secondDiagonal.count == 2 &&
              secondDiagonal.values[0].position.x * secondDiagonal.values[0].position.y < 0.0F &&
              secondDiagonal.values[1].position.x * secondDiagonal.values[1].position.y < 0.0F,
          "Second beacon pair occupies the opposite diagonal");
    check(vkexp::beaconPhaseForStep(vkexp::BeaconScenario::AlternatingDiagonals, 449, 900) == 0 &&
              vkexp::beaconPhaseForStep(vkexp::BeaconScenario::AlternatingDiagonals, 450, 900) == 1,
          "Beacon diagonal changes at the generation midpoint");

    settings.beaconScenario = vkexp::BeaconScenario::Rotating;
    settings.beaconRadiusRatio = std::hypot(1.0F, 0.5F) / settings.worldRadius;
    settings.beaconRotationAngle = 1.57079632679F;
    const vkexp::ActiveBeacons rotating = vkexp::activeBeacons(agent, settings);
    check(rotating.count == 1 && closeTo(rotating.values[0].position.x, -0.5F) &&
              closeTo(rotating.values[0].position.y, 1.0F),
          "Rotating beacon orbits around the world center");
    check(closeTo(vkexp::beaconRotationAngleForStep(0.5F, 0.1F, 20), 1.0F),
          "Beacon angle follows simulation time and angular speed");

    settings.beaconScenario = vkexp::BeaconScenario::RandomMovement;
    settings.beaconRadiusRatio = 0.65F;
    settings.beaconTeleportProbability = 0.25F;
    settings.beaconMotionSeed = 42U;
    settings.beaconMotionTime = 0.0F;
    const vkexp::ActiveBeacons randomStart = vkexp::activeBeacons(agent, settings);
    settings.beaconRandomSpeed = 0.0F;
    settings.beaconMotionTime = 1.5F;
    const vkexp::ActiveBeacons randomStopped = vkexp::activeBeacons(agent, settings);
    settings.beaconRandomSpeed = 0.18F;
    const vkexp::ActiveBeacons randomMiddle = vkexp::activeBeacons(agent, settings);
    const float maximumRoamRadius = settings.worldRadius * settings.beaconRadiusRatio;
    check(randomStart.count == 1 &&
              std::hypot(randomStart.values[0].position.x, randomStart.values[0].position.y) <=
                  maximumRoamRadius &&
              std::hypot(randomMiddle.values[0].position.x, randomMiddle.values[0].position.y) <=
                  maximumRoamRadius,
          "Random beacon remains inside its configured roaming radius");
    check(closeTo(randomStart.values[0].position.x, randomStopped.values[0].position.x) &&
              closeTo(randomStart.values[0].position.y, randomStopped.values[0].position.y),
          "Zero wander speed stops continuous random movement");
    check(!closeTo(randomStart.values[0].position.x, randomMiddle.values[0].position.x) ||
              !closeTo(randomStart.values[0].position.y, randomMiddle.values[0].position.y),
          "Random beacon moves between deterministic waypoints");

    settings.beaconScenario = vkexp::BeaconScenario::ForageHome;
    settings.beaconRotationAngle = 0.0F;
    settings.beaconMotionSeed = 42U;
    settings.beaconMotionTime = 0.0F;
    const vkexp::ActiveBeacons forage = vkexp::activeBeacons(agent, settings);
    check(forage.count == 2 && forage.values[0].color.x > forage.values[0].color.z &&
              forage.values[1].color.z > forage.values[1].color.x,
          "Forage scenario exposes orange resource and blue home beacons");
    agent.pose = forage.values[0].position;
    agent.internal.y = 0.0F;
    check(closeTo(vkexp::nearestBeaconDistance(agent, settings), 0.0F),
          "Forage task targets the resource while empty");
    agent.pose = forage.values[1].position;
    agent.internal.y = 1.0F;
    check(closeTo(vkexp::nearestBeaconDistance(agent, settings), 0.0F),
          "Forage task targets home while carrying cargo");
    settings.beaconMotionTime = vkexp::forageHomeRelocationSeconds - 0.01F;
    const vkexp::Float4 homeBeforeRelocation = vkexp::homeBeaconPosition(agent, settings);
    settings.beaconMotionTime = vkexp::forageHomeRelocationSeconds;
    const vkexp::Float4 homeAfterRelocation = vkexp::homeBeaconPosition(agent, settings);
    check(!closeTo(homeBeforeRelocation.x, homeAfterRelocation.x) ||
              !closeTo(homeBeforeRelocation.y, homeAfterRelocation.y),
          "Forage home deterministically relocates at its configured interval");
    check(vkexp::homeBeaconRelocated(settings),
          "Forage scenario reports the exact home relocation step");
    settings.beaconMotionTime += settings.deltaTime;
    check(!vkexp::homeBeaconRelocated(settings),
          "Forage scenario reports relocation for only one simulation step");
}

void testForageCycleAndMemory() {
    vkexp::AgentState agent{};
    agent.pose.w = 0.022F;
    agent.motion.w = 1.0F;
    agent.target = {1.0F, 0.0F, 0.0F, 0.0F};
    vkexp::SimulationStep settings{};
    settings.beaconScenario = vkexp::BeaconScenario::ForageHome;
    settings.beaconRotationAngle = 0.0F;
    const vkexp::ActiveBeacons beacons = vkexp::activeBeacons(agent, settings);
    agent.pose.x = beacons.values[0].position.x;
    agent.pose.y = beacons.values[0].position.y;
    agent.metrics = {};

    vkexp::neuro::Weights weights{};
    constexpr std::size_t outputBias =
        vkexp::neuro::Topology::inputCount * vkexp::neuro::Topology::hiddenCount +
        vkexp::neuro::Topology::hiddenCount +
        vkexp::neuro::Topology::hiddenCount * vkexp::neuro::Topology::outputCount;
    weights[outputBias + 6] = 0.5F;
    weights[outputBias + 7] = -0.75F;

    vkexp::stepAgentCpu(agent, weights, settings);
    check(closeTo(agent.internal.x, 1.0F) && closeTo(agent.internal.y, 1.0F),
          "Resource pickup fills cargo and switches the task to home");
    check(closeTo(agent.internal.z, std::tanh(0.5F)) &&
              closeTo(agent.internal.w, std::tanh(-0.75F)),
          "Neural memory outputs persist in agent state");

    agent.pose.x = beacons.values[1].position.x;
    agent.pose.y = beacons.values[1].position.y;
    agent.motion.x = 0.0F;
    agent.motion.y = 0.0F;
    vkexp::stepAgentCpu(agent, weights, settings);
    check(closeTo(agent.internal.x, 0.0F) && closeTo(agent.internal.y, 0.0F),
          "Home delivery empties cargo and switches the task back to resource");
    check(vkexp::completedForageCycles(agent) == 1,
          "Home delivery completes one forage cycle");
    check(vkexp::agentFitness(agent, settings.beaconScenario) > 2.0F,
          "Completed forage cycle produces positive fitness");

    vkexp::AgentState radiusProbe{};
    radiusProbe.pose = {beacons.values[0].position.x + vkexp::beaconVisualRadius * 2.0F,
                        beacons.values[0].position.y, 0.0F, 0.022F};
    radiusProbe.motion.w = 1.0F;
    radiusProbe.target = {1.0F, 0.0F, 0.0F, 0.0F};
    radiusProbe.metrics = {vkexp::beaconVisualRadius * 2.0F,
                           vkexp::beaconVisualRadius * 2.0F, 0.0F, 0.0F};
    vkexp::neuro::Weights zeroWeights{};
    settings.arrivalRadiusMultiplier = 1.0F;
    vkexp::stepAgentCpu(radiusProbe, zeroWeights, settings);
    check(radiusProbe.internal.y < 0.5F,
          "Default arrival radius requires entering the beacon circle");
    settings.arrivalRadiusMultiplier = 2.1F;
    vkexp::stepAgentCpu(radiusProbe, zeroWeights, settings);
    check(radiusProbe.internal.y >= 0.5F,
          "Expanded arrival radius permits pickup outside the visible circle");
}

void testWallCollisionPenalty() {
    vkexp::AgentState agent{};
    agent.pose = {1.817F, 0.0F, 0.0F, 0.022F};
    agent.motion = {0.55F, 0.0F, 0.0F, 1.0F};
    agent.target = {0.0F, 0.0F, 0.0F, 0.0F};
    agent.metrics = {1.817F, 1.817F, 0.0F, 0.0F};
    vkexp::SimulationStep settings{};
    settings.worldShape = vkexp::WorldShape::Square;
    settings.wallCollisionPenalty = 0.1F;
    const vkexp::neuro::Weights weights{};

    vkexp::stepAgentCpu(agent, weights, settings);

    const float touch = agent.wallTouch0.x + agent.wallTouch0.y + agent.wallTouch0.z +
                        agent.wallTouch0.w + agent.wallTouch1.x + agent.wallTouch1.y +
                        agent.wallTouch1.z + agent.wallTouch1.w;
    check(touch > 0.0F, "World boundary produces tactile contact");
    check(agent.penalties.x > 0.0F, "World boundary accumulates a fitness penalty");
    check(vkexp::agentFitness(agent) <
              agent.metrics.w + (agent.metrics.x - agent.metrics.y) - agent.metrics.z * 0.002F,
          "Wall collision penalty lowers fitness");
}

void testGeneticAlgorithm() {
    const vkexp::EvolutionSettings settings{8, 2, 3, 0.5F, 0.1F, 0.2F, 42U};
    vkexp::GeneticAlgorithm evolution{settings};
    const std::vector<vkexp::Genome> original = evolution.population();
    const std::vector<float> fitness{-4.0F, -3.0F, -2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F};
    const vkexp::GenerationSummary summary = evolution.evolve(fitness);
    check(summary.championIndex == 7, "GA champion selection");
    check(closeTo(summary.bestFitness, 3.0F), "GA best fitness");
    check(evolution.generation() == 1, "GA generation counter");
    check(evolution.population().front().weights == original.back().weights,
          "GA preserves champion as first elite");
}

} // namespace

int main() {
    testTimingSeries();
    testCpuProfiler();
    testDispatchSize();
    testComputeResourceValidation();
    testLogicalWorldPartition();
    testPingPongState();
    testNeuralNetworkContract();
    testMultimodalSensors();
    testWorldAndBeaconScenarios();
    testForageCycleAndMemory();
    testWallCollisionPenalty();
    testGeneticAlgorithm();
    return failures == 0 ? 0 : 1;
}
