#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/evolution/GenomeArchive.hpp"
#include "vkexp/neuro/BrainKernel.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/profiling/CpuProfiler.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/CpuSimulation.hpp"
#include "vkexp/simulation/Sensors.hpp"
#include "vkexp/simulation/TrailKernel.hpp"
#include "vkexp/simulation/Units.hpp"
#include "vkexp/worlds/ScenarioKernel.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
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
    // Derived from the preset rather than pinned to a snapshot: adding a sensor
    // is meant to be a one-line edit in BrainKernel.inl, not a test rewrite.
    namespace kernel = vkexp::neuro::kernel;
    check(vkexp::neuro::Topology::inputCount ==
              kernel::BrainLightReceptorCount * kernel::BrainLightChannels +
                  kernel::BrainTactileSectorCount * kernel::BrainTactileChannels +
                  kernel::BrainAntennaCount * kernel::BrainAntennaChannels +
                  kernel::BrainSelfInputCount + kernel::BrainTaskInputCount +
                  kernel::BrainRecurrentCount,
          "Input capacity is the sum of the declared sensor blocks");
    check(vkexp::neuro::Topology::outputCount ==
              kernel::BrainActuatorOutputCount + kernel::BrainRecurrentCount,
          "Output capacity is actuators plus recurrent cells");
    check(vkexp::neuro::Topology::weightCount == vkexp::neuro::maximumBrainShape.weightCount(),
          "Genome capacity matches the widest brain shape");

    // The sensor blocks must tile the input vector without gaps or overlaps.
    check(kernel::BrainLightOffset == 0, "Light block starts the input vector");
    check(kernel::brainLightChannelIndex(kernel::BrainLightReceptorCount - 1,
                                         kernel::BrainLightChannels - 1) +
                  1 ==
              kernel::BrainTactileOffset,
          "Tactile block follows the light block");
    check(kernel::brainTactileChannelIndex(kernel::BrainTactileSectorCount - 1,
                                           kernel::BrainTactileChannels - 1) +
                  1 ==
              kernel::BrainAntennaOffset,
          "Antenna block follows the tactile block");
    check(kernel::brainAntennaChannelIndex(kernel::BrainAntennaCount - 1,
                                           kernel::BrainAntennaChannels - 1) +
                  1 ==
              kernel::BrainSelfOffset,
          "Self block follows the antenna block");
    // The antennae have to reach into different trail cells or the three
    // readings collapse into one number and carry no gradient.
    const float antennaSpread =
        2.0F * kernel::BrainAntennaLength * std::sin(kernel::BrainAntennaHalfSpread);
    // Strictly more than one cell apart is the guarantee that matters: two points
    // further apart than a cell is wide cannot share a cell, whatever the phase.
    // At the coarsest 8 cm setting the 15.5 cm spread leaves 1.9 cells.
    check(antennaSpread > vkexp::trailCellSizeForBodyFraction(vkexp::trailCellFractionCoarsest),
          "Outer antenna tips cannot share a cell at the coarsest trail resolution");
    check(kernel::BrainSelfOffset + kernel::BrainSelfInputCount == kernel::BrainTaskOffset,
          "Task block follows the self block");
    check(kernel::BrainTaskOffset + kernel::BrainTaskInputCount ==
              kernel::BrainRecurrentInputOffset,
          "Recurrent inputs follow the task block");
    check(kernel::BrainRecurrentInputOffset + kernel::BrainRecurrentCount ==
              kernel::BrainInputCapacity,
          "Recurrent inputs close the input vector");

    const auto& stationary = vkexp::scenarioDefinition(vkexp::BeaconScenario::Stationary);
    const auto& forage = vkexp::scenarioDefinition(vkexp::BeaconScenario::ForageHome);
    check(stationary.brain.inputCount == vkexp::neuro::Topology::inputCount -
                                             vkexp::neuro::Topology::taskInputCount -
                                             vkexp::neuro::Topology::recurrentMemoryCount &&
              stationary.brain.outputCount == vkexp::neuro::Topology::actuatorOutputCount,
          "Stationary scenario owns a reactive brain shape without task or memory");
    check(forage.brain.inputCount == vkexp::neuro::Topology::inputCount &&
              forage.brain.outputCount == vkexp::neuro::Topology::outputCount,
          "Forage scenario owns the full recurrent brain shape");
    check(std::string_view(stationary.name) == "Stationary" &&
              std::string_view(forage.name) == "Forage + home",
          "Scenario definitions own their display names");
    const std::uint32_t layout = vkexp::neuro::packBrainLayout(stationary.brain);
    const vkexp::neuro::BrainShape unpacked = vkexp::neuro::brainShape(layout);
    check(vkexp::neuro::brainGenomeStride(layout) == vkexp::neuro::Topology::weightCount &&
              unpacked.inputCount == stationary.brain.inputCount &&
              unpacked.hiddenCount == stationary.brain.hiddenCount &&
              unpacked.outputCount == stationary.brain.outputCount,
          "Packed GPU brain layout preserves capacity and active shape");
}

void testMultimodalSensors() {
    vkexp::AgentState agent{};
    agent.pose = {0.0F, 0.0F, 0.0F, vkexp::agentBodyRadius};
    agent.motion.w = 1.0F;
    agent.target = {1.0F, 0.0F, 0.0F, 0.0F};
    agent.internal = {0.7F, 1.0F, -0.4F, 0.6F};
    agent.wallTouch0.x = 0.7F;
    agent.agentTouch1.w = 0.8F;
    const vkexp::SimulationStep settings{};
    const vkexp::neuro::Inputs inputs = vkexp::sampleAgentInputs(agent, settings);
    // Offsets come from the preset, not from arithmetic repeated here: the point
    // of testNeuralNetworkContract is that the preset derives them correctly, so
    // restating the sums would only test the restatement.
    namespace topology = vkexp::neuro;
    constexpr std::size_t center = 3 * topology::Topology::lightChannelsPerReceptor;
    check(inputs[center] > 0.0F && inputs[center + 1] > inputs[center],
          "RGB photoreceptor observes cyan beacon");
    check(inputs[center + 3] > 0.0F, "Photoreceptor luminance channel");
    constexpr std::size_t tactile = topology::Topology::tactileOffset;
    check(closeTo(inputs[tactile], 0.7F), "Wall tactile sector mapping");
    check(closeTo(inputs[tactile + 7 * 2 + 1], 0.8F), "Agent tactile sector mapping");
    constexpr std::size_t task = topology::Topology::taskOffset;
    check(closeTo(inputs[task], 0.7F) && closeTo(inputs[task + 1], 1.0F),
          "Cargo and task-state input mapping");
    constexpr std::size_t recurrent = topology::Topology::recurrentInputOffset;
    check(closeTo(inputs[recurrent], -0.4F) && closeTo(inputs[recurrent + 1], 0.6F),
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
    check(closeTo(vkexp::beaconArrivalRadius(arrivalSettings), vkexp::beaconVisualRadius * 0.1F),
          "Minimum arrival multiplier scales the beacon radius");
    arrivalSettings.arrivalRadiusMultiplier = 5.0F;
    check(closeTo(vkexp::beaconArrivalRadius(arrivalSettings), vkexp::beaconVisualRadius * 5.0F),
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
    agent.pose.w = vkexp::agentBodyRadius;
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
    check(vkexp::completedForageCycles(agent) == 1, "Home delivery completes one forage cycle");
    check(vkexp::agentFitness(agent, settings.beaconScenario) > 2.0F,
          "Completed forage cycle produces positive fitness");

    vkexp::AgentState radiusProbe{};
    radiusProbe.pose = {beacons.values[0].position.x + vkexp::beaconVisualRadius * 2.0F,
                        beacons.values[0].position.y, 0.0F, vkexp::agentBodyRadius};
    radiusProbe.motion.w = 1.0F;
    radiusProbe.target = {1.0F, 0.0F, 0.0F, 0.0F};
    radiusProbe.metrics = {vkexp::beaconVisualRadius * 2.0F, vkexp::beaconVisualRadius * 2.0F, 0.0F,
                           0.0F};
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
    agent.pose = {1.817F, 0.0F, 0.0F, vkexp::agentBodyRadius};
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
    const vkexp::FitnessWeights fitnessWeights{};
    check(vkexp::agentFitness(agent, settings.beaconScenario, fitnessWeights) <
              agent.metrics.w + (agent.metrics.x - agent.metrics.y) -
                  agent.metrics.z * fitnessWeights.motorCostWeight,
          "Wall collision penalty lowers fitness");
}

// The step used to mix two time bases: velocities, drags and motor costs were
// integrated per second, while the wall penalty and the contact solver
// accumulated per step. That made deltaTime a fitness parameter in disguise --
// the same trial at 240 Hz charged four times the wall penalty of one at 60 Hz,
// which is why deltaTime was pinned at 1/60 and never exposed.
//
// Driving an agent into a wall and holding it there for two simulated seconds
// is what makes the difference visible. Charging per second leaves a residual
// spread of about 1.4x across a 16x change in step rate, all of it from contact
// detection in a bouncing model: the impact speed feeding the contact strength
// shrinks with the step, and the chatter duty cycle drifts from 95% to 88%.
// Charging per step would instead scale straight with the step count.
void testFixedStepIndependence() {
    const vkexp::neuro::BrainShape brain =
        vkexp::scenarioDefinition(vkexp::BeaconScenario::Stationary).brain;
    const auto inputCount = static_cast<vkexp::neuro::kernel::uint>(brain.inputCount);
    const auto hiddenCount = static_cast<vkexp::neuro::kernel::uint>(brain.hiddenCount);
    const auto outputCount = static_cast<vkexp::neuro::kernel::uint>(brain.outputCount);
    vkexp::neuro::Weights drivingWeights{};
    for (const vkexp::neuro::kernel::uint motor : {vkexp::neuro::kernel::BrainMotorLeftOutput,
                                                   vkexp::neuro::kernel::BrainMotorRightOutput}) {
        drivingWeights[vkexp::neuro::kernel::brainOutputBiasIndex(0, inputCount, hiddenCount,
                                                                  outputCount, motor)] = 3.0F;
    }

    const auto penaltyForTwoSeconds = [&drivingWeights](const float rateHz) {
        vkexp::SimulationStep settings{};
        settings.deltaTime = 1.0F / rateHz;
        settings.worldShape = vkexp::WorldShape::Square;
        settings.wallCollisionPenalty = 0.6F;
        vkexp::AgentState agent{};
        agent.pose = {1.81F, 0.0F, 0.0F, vkexp::agentBodyRadius};
        agent.motion = {0.0F, 0.0F, 0.0F, 1.0F};
        agent.metrics = {1.81F, 1.81F, 0.0F, 0.0F};
        const auto steps = static_cast<std::uint32_t>(rateHz * 2.0F);
        for (std::uint32_t step = 0; step < steps; ++step) {
            vkexp::stepAgentCpu(agent, drivingWeights, settings);
        }
        return agent.penalties.x;
    };

    const float baseline = penaltyForTwoSeconds(vkexp::units::simulationRateHz);
    check(baseline > 0.0F, "Driving into a wall for two seconds accumulates a penalty");
    for (const float rateHz : {30.0F, 120.0F, 240.0F, 480.0F}) {
        const float ratio = penaltyForTwoSeconds(rateHz) / baseline;
        const float stepCountRatio = rateHz / vkexp::units::simulationRateHz;
        check(ratio > 0.75F && ratio < 1.25F,
              "Wall penalty over two simulated seconds barely moves with the step rate");
        // The discriminating half: per-step accumulation would put the ratio at
        // the step-count ratio instead, which is 0.5x to 8x here.
        check(std::abs(ratio - 1.0F) < std::abs(stepCountRatio - 1.0F) * 0.5F,
              "Wall penalty tracks simulated time rather than step count");
    }

    check(std::abs(vkexp::units::secondsForSteps(900, vkexp::units::fixedTimeStep) - 15.0F) < 1e-5F,
          "900 steps at the fixed rate is 15 seconds");
    check(std::abs(vkexp::units::secondsForSteps(3600, 1.0F / 240.0F) - 15.0F) < 1e-5F,
          "3600 steps at 240 Hz covers the same 15 seconds");
}

void testScenarioRegistryContract() {
    const std::span<const vkexp::ScenarioDefinition* const> registry = vkexp::scenarioRegistry();
    check(registry.size() == vkexp::beaconScenarioCount, "Registry covers every BeaconScenario");
    for (std::size_t index = 0; index < registry.size(); ++index) {
        const vkexp::ScenarioDefinition& scenario = *registry[index];
        const std::string label{scenario.name};
        check(scenario.id == static_cast<vkexp::BeaconScenario>(index),
              label + ": registry order matches the enum");
        check(&vkexp::scenarioDefinition(scenario.id) == &scenario,
              label + ": lookup returns the registered definition");
        check(scenario.key != nullptr && scenario.key[0] != '\0', label + ": has a CLI key");
        check(scenario.brain.fitsCapacity(), label + ": brain fits the genome capacity");

        // beaconCount is declared separately because the renderer needs it
        // without an agent; it must still agree with what beacons() reports.
        vkexp::SimulationStep settings{};
        settings.beaconScenario = scenario.id;
        vkexp::AgentState agent{};
        agent.target = {settings.worldRadius * 0.7F, 0.0F, 0.0F, 0.0F};
        check(scenario.beacons(agent, settings).count == scenario.beaconCount,
              label + ": declared beacon count matches the beacons it reports");

        // Every scenario must be steppable without the caller knowing which it is.
        const vkexp::neuro::Weights zeroWeights{};
        vkexp::AgentState stepped = agent;
        stepped.pose.w = vkexp::agentBodyRadius;
        vkexp::stepAgentCpu(stepped, zeroWeights, settings);
        check(std::isfinite(stepped.pose.x) && std::isfinite(stepped.metrics.w),
              label + ": one step through the hooks stays finite");
    }
}

void testFitnessWeightsAreParameters() {
    vkexp::AgentState agent{};
    agent.metrics = {2.0F, 1.0F, 10.0F, 3.0F};
    agent.target.w = 1.0F; // one completed phase
    const vkexp::BeaconScenario scenario = vkexp::BeaconScenario::Stationary;

    vkexp::FitnessWeights base{};
    const float reference = vkexp::agentFitness(agent, scenario, base);

    vkexp::FitnessWeights doubledBonus = base;
    doubledBonus.objectiveBonus = base.objectiveBonus * 2.0F;
    check(closeTo(vkexp::agentFitness(agent, scenario, doubledBonus),
                  reference + base.objectiveBonus),
          "Objective bonus is a parameter, not a literal");

    vkexp::FitnessWeights freeMotors = base;
    freeMotors.motorCostWeight = 0.0F;
    check(closeTo(vkexp::agentFitness(agent, scenario, freeMotors),
                  reference + agent.metrics.z * base.motorCostWeight),
          "Motor cost weight is a parameter, not a literal");
}

void testSharedScenarioKernel() {
    namespace kernel = vkexp::worlds::kernel;
    // The shaders compile these same functions from ScenarioKernel.inl, and the
    // CPU/GPU parity tests compare the results; these checks pin the C++ side so
    // a change to the shared source cannot pass unnoticed without a GPU.
    check(kernel::scenarioHash(0U) == 0U, "Hash of zero is zero");
    const float sample = kernel::scenarioRandom01(12345U);
    check(sample >= 0.0F && sample <= 1.0F, "Scenario random is normalised");
    check(closeTo(kernel::scenarioRandom01(12345U), sample), "Scenario random is deterministic");

    check(kernel::forageHomeEpoch(0.0F) == 0U, "Forage epoch starts at zero");
    check(kernel::forageHomeEpoch(kernel::ForageHomeRelocationSeconds + 0.1F) == 1U,
          "Forage epoch advances at the relocation period");
    check(kernel::forageHomeRelocated(kernel::ForageHomeRelocationSeconds, 1.0F / 60.0F),
          "Forage relocation is reported on the epoch boundary");
    check(!kernel::forageHomeRelocated(1.0F, 1.0F / 60.0F),
          "Forage relocation is not reported mid-epoch");

    const kernel::vec2 home = kernel::forageHomeOffset(kernel::forageHomeKey(7U, 1U, 2U));
    const float homeRatio = kernel::length(home);
    check(homeRatio >= kernel::ForageHomeMinimumRadiusRatio - 0.0001F &&
              homeRatio <=
                  kernel::ForageHomeMinimumRadiusRatio + kernel::ForageHomeRadiusRange + 0.0001F,
          "Forage home stays inside its configured annulus");

    // A quarter turn maps +x to +y.
    const kernel::vec2 rotated =
        kernel::rotatingOrbitOffset({1.0F, 0.0F}, kernel::ScenarioTau * 0.25F);
    check(closeTo(rotated.x, 0.0F) && closeTo(rotated.y, 1.0F), "Orbit offset rotates correctly");

    const kernel::vec2 wander =
        kernel::randomWanderOffset(kernel::randomWanderKey(3U, 0U, 0U), 5.0F);
    check(kernel::length(wander) <= 1.0F, "Wander offset stays within the roam radius");

    check(kernel::alternatingDiagonalOffset(0U, 0U).x < 0.0F &&
              kernel::alternatingDiagonalOffset(1U, 0U).x > 0.0F,
          "Alternating beacons sit on opposite sides");
}

void testGenomeArchiveRoundTrip() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "vkexp_archive_test" / "population.vkng";
    std::vector<vkexp::Genome> genomes(3);
    for (std::size_t index = 0; index < genomes.size(); ++index) {
        for (std::size_t weight = 0; weight < genomes[index].weights.size(); ++weight) {
            genomes[index].weights[weight] =
                std::sin(static_cast<float>(index * 31 + weight) * 0.017F);
        }
    }
    const vkexp::GenomeArchiveMetadata metadata{42, 4, 0xC0FFEEU, 1.5F, 0.25F, 52, 20, 8};
    vkexp::saveGenomeArchive(path, genomes, metadata);

    const vkexp::GenomeArchive loaded = vkexp::loadGenomeArchive(path);
    check(loaded.genomes.size() == genomes.size(), "Archive genome count round-trip");
    check(loaded.metadata.generation == 42, "Archive generation round-trip");
    check(loaded.metadata.scenario == 4, "Archive scenario round-trip");
    check(loaded.metadata.seed == 0xC0FFEEU, "Archive seed round-trip");
    check(closeTo(loaded.metadata.bestFitness, 1.5F), "Archive best fitness round-trip");
    check(loaded.metadata.brainOutputCount == 8, "Archive brain shape round-trip");
    bool identical = true;
    for (std::size_t index = 0; index < genomes.size(); ++index) {
        identical = identical && loaded.genomes[index].weights == genomes[index].weights;
    }
    check(identical, "Archive weights round-trip bit-exactly");

    // A corrupted magic must fail loudly rather than load noise as a population.
    const std::filesystem::path corrupted = path.parent_path() / "corrupted.vkng";
    std::filesystem::copy_file(path, corrupted, std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream stream{corrupted, std::ios::binary | std::ios::in | std::ios::out};
        stream.seekp(0);
        stream.write("XXXX", 4);
    }
    bool rejectedMagic = false;
    try {
        (void)vkexp::loadGenomeArchive(corrupted);
    } catch (const vkexp::GenomeArchiveError&) {
        rejectedMagic = true;
    }
    check(rejectedMagic, "Archive rejects a foreign file");

    // Truncation must not yield a half-filled population either.
    const std::filesystem::path truncated = path.parent_path() / "truncated.vkng";
    std::filesystem::copy_file(path, truncated, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 16);
    bool rejectedTruncation = false;
    try {
        (void)vkexp::loadGenomeArchive(truncated);
    } catch (const vkexp::GenomeArchiveError&) {
        rejectedTruncation = true;
    }
    check(rejectedTruncation, "Archive rejects a truncated file");

    std::error_code cleanupError;
    std::filesystem::remove_all(path.parent_path(), cleanupError);
}

void testPopulationReload() {
    const vkexp::EvolutionSettings settings{8, 2, 3, 0.5F, 0.1F, 0.2F, 42U};
    vkexp::GeneticAlgorithm evolution{settings};
    std::vector<vkexp::Genome> replacement(settings.populationSize);
    replacement.front().weights[0] = 3.25F;
    evolution.setPopulation(replacement, 17);
    check(evolution.generation() == 17, "Loaded population restores the generation counter");
    check(closeTo(evolution.population().front().weights[0], 3.25F),
          "Loaded population replaces the weights");

    bool rejectedMismatch = false;
    try {
        const std::vector<vkexp::Genome> wrongSize(settings.populationSize - 1);
        evolution.setPopulation(wrongSize, 0);
    } catch (const std::invalid_argument&) {
        rejectedMismatch = true;
    }
    check(rejectedMismatch, "Loaded population size mismatch rejection");
}

void testStepParameterPacking() {
    // The GPU step parameters outgrew the 128 bytes Vulkan guarantees for push
    // constants, which is why they now travel in a storage buffer.
    check(sizeof(vkexp::GpuStepParameters) > 128,
          "Step parameters exceed the guaranteed push constant size");
    check(sizeof(vkexp::ScenarioParameterBlock) == 48, "Scenario block size");

    vkexp::SimulationStep settings{};
    settings.beaconScenario = vkexp::BeaconScenario::Rotating;
    settings.beaconRotationAngle = 0.75F;
    settings.beaconRadiusRatio = 0.6F;
    const vkexp::ScenarioParameterBlock rotating =
        vkexp::scenarioDefinition(settings.beaconScenario).gpuParameters(settings);
    check(closeTo(rotating.floats0.x, 0.75F), "Rotating scenario packs its rotation angle");
    check(closeTo(rotating.floats0.y, 0.6F), "Rotating scenario packs its radius ratio");

    settings.beaconScenario = vkexp::BeaconScenario::ForageHome;
    settings.beaconMotionSeed = 0xABCDU;
    settings.forageCargoDecayRate = 0.11F;
    const vkexp::ScenarioParameterBlock forage =
        vkexp::scenarioDefinition(settings.beaconScenario).gpuParameters(settings);
    check(forage.integers[0] == 0xABCDU, "Forage scenario packs its motion seed");
    check(closeTo(forage.floats0.w, 0.11F), "Forage scenario packs its cargo decay rate");

    // Stationary beacons live in the agent, so its block must stay empty.
    settings.beaconScenario = vkexp::BeaconScenario::Stationary;
    const vkexp::ScenarioParameterBlock stationary =
        vkexp::scenarioDefinition(settings.beaconScenario).gpuParameters(settings);
    check(stationary.floats0.x == 0.0F && stationary.integers[0] == 0U,
          "Stationary scenario sends no scenario parameters");
}

void testResolvedStepSettings() {
    vkexp::SimulationStep base{};
    base.beaconScenario = vkexp::BeaconScenario::AlternatingDiagonals;
    base.beaconAngularSpeed = 1.0F;
    constexpr std::uint32_t steps = 100;
    const vkexp::SimulationStep before = vkexp::resolveStepSettings(base, 10, steps);
    const vkexp::SimulationStep atFlip = vkexp::resolveStepSettings(base, steps / 2, steps);
    const vkexp::SimulationStep after = vkexp::resolveStepSettings(base, steps / 2 + 1, steps);
    check(before.beaconPhase == 0 && !before.beaconPhaseChanged, "Phase before the flip");
    check(atFlip.beaconPhase == 1 && atFlip.beaconPhaseChanged, "Phase change reported once");
    check(after.beaconPhase == 1 && !after.beaconPhaseChanged, "Phase after the flip");
    check(closeTo(atFlip.beaconMotionTime, base.deltaTime * static_cast<float>(steps / 2)),
          "Resolved motion time follows the step index");
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
    testFixedStepIndependence();
    testMultimodalSensors();
    testWorldAndBeaconScenarios();
    testForageCycleAndMemory();
    testWallCollisionPenalty();
    testGeneticAlgorithm();
    testScenarioRegistryContract();
    testFitnessWeightsAreParameters();
    testSharedScenarioKernel();
    testGenomeArchiveRoundTrip();
    testPopulationReload();
    testStepParameterPacking();
    testResolvedStepSettings();
    return failures == 0 ? 0 : 1;
}
