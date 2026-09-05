#include "vkexp/simulation/SimulationDriver.hpp"

#include "vkexp/simulation/CpuSimulation.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>

namespace vkexp {
namespace {

constexpr VkMemoryPropertyFlags hostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

struct alignas(16) GridBuildParameters {
    float worldRadius{};
    float cellSize{};
    std::uint32_t agentCount{};
    std::uint32_t worldCount{};
    std::uint32_t gridWidth{};
    std::uint32_t gridCellsPerWorld{};
    std::uint32_t reserved0{};
    std::uint32_t reserved1{};
};
static_assert(sizeof(GridBuildParameters) == 32);

void createStorageLayout(const VkDevice device, const std::uint32_t bindingCount,
                         UniqueDescriptorSetLayout& layout) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
    for (std::uint32_t binding = 0; binding < bindingCount; ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[binding].descriptorCount = 1;
        bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = bindingCount;
    info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &info, nullptr, layout.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create agent compute descriptor layout");
    }
}

} // namespace

SimulationDriver::SimulationDriver(SimulationState& state, EvolutionSettings evolution,
                                   SimulationDriverConfig config)
    : state_(state), evolution_(evolution), config_(config) {
    if (config_.trialsPerGenome == 0 || config_.maximumStepsPerBatch == 0 ||
        config_.gridCellSize <= 0.0F) {
        throw std::invalid_argument("Invalid simulation driver configuration");
    }
    state_.evolution = evolution_.settings();
}

void SimulationDriver::createResources(const VkPhysicalDevice physicalDevice,
                                       const VkDevice device) {
    physicalDevice_ = physicalDevice;
    device_ = device;
    createStepResources();
    resetGeneration();
}

void SimulationDriver::createStepResources() {
    const auto genomeCount = static_cast<std::uint32_t>(evolution_.population().size());
    const std::uint32_t agentCount = genomeCount * config_.trialsPerGenome;
    const VkDeviceSize agentBytes = sizeof(AgentState) * agentCount;
    const VkDeviceSize genomeBytes = sizeof(float) * neuro::Topology::weightCount * genomeCount;
    const VkDeviceSize stepParameterBytes =
        sizeof(GpuStepParameters) * config_.maximumStepsPerBatch;
    agentBuffers_.create(physicalDevice_, device_,
                         {agentBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    genomeBuffer_.create(physicalDevice_, device_,
                         {genomeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    stepParameterBuffer_.create(
        physicalDevice_, device_,
        {stepParameterBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    stepParameterStaging_.resize(config_.maximumStepsPerBatch);

    updateWorldLayout();
    updateGridDimensions();
    ensureGridCapacity();
    const VkDeviceSize gridLinkBytes = sizeof(std::int32_t) * agentCount;
    gridNext_.create(physicalDevice_, device_, {gridLinkBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});

    createStorageLayout(device_, 6, stepDescriptorSetLayout_);
    createStorageLayout(device_, 1, gridClearDescriptorSetLayout_);
    createStorageLayout(device_, 3, gridBuildDescriptorSetLayout_);
    descriptorAllocator_.create(device_, {5, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 19}}});

    for (std::uint32_t readIndex = 0; readIndex < 2; ++readIndex) {
        const VkBuffer readBuffer =
            readIndex == 0 ? agentBuffers_.read().buffer() : agentBuffers_.write().buffer();
        const VkBuffer writeBuffer =
            readIndex == 0 ? agentBuffers_.write().buffer() : agentBuffers_.read().buffer();
        stepDescriptorSets_[readIndex] =
            descriptorAllocator_.allocate(stepDescriptorSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, readBuffer, 0, agentBytes)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, writeBuffer, 0, agentBytes)
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, genomeBuffer_.buffer(), 0,
                         genomeBuffer_.size())
            .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                         gridHeads_.size())
            .writeBuffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridNext_.buffer(), 0,
                         gridNext_.size())
            .writeBuffer(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stepParameterBuffer_.buffer(), 0,
                         stepParameterBuffer_.size())
            .update(device_, stepDescriptorSets_[readIndex]);

        gridBuildDescriptorSets_[readIndex] =
            descriptorAllocator_.allocate(gridBuildDescriptorSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, readBuffer, 0, agentBytes)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                         gridHeads_.size())
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridNext_.buffer(), 0,
                         gridNext_.size())
            .update(device_, gridBuildDescriptorSets_[readIndex]);
    }
    gridClearDescriptorSet_ = descriptorAllocator_.allocate(gridClearDescriptorSetLayout_.get());
    DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                     gridHeads_.size())
        .update(device_, gridClearDescriptorSet_);

    stepPipeline_ = ComputePipelineBuilder{physicalDevice_, device_}
                        .shader(VKEXP_SHADER_DIR "/agent_step.comp.spv")
                        .addDescriptorSetLayout(stepDescriptorSetLayout_.get())
                        .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
                        .build();
    gridClearPipeline_ =
        ComputePipelineBuilder{physicalDevice_, device_}
            .shader(VKEXP_SHADER_DIR "/agent_grid_clear.comp.spv")
            .addDescriptorSetLayout(gridClearDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
            .build();
    gridBuildPipeline_ =
        ComputePipelineBuilder{physicalDevice_, device_}
            .shader(VKEXP_SHADER_DIR "/agent_grid_build.comp.spv")
            .addDescriptorSetLayout(gridBuildDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(GridBuildParameters))
            .build();

    state_.agents = {{agentBuffers_.read().buffer(), agentBuffers_.write().buffer()},
                     agentBytes,
                     agentBuffers_.readIndex(),
                     agentCount,
                     genomeCount,
                     config_.trialsPerGenome,
                     0};
}

void SimulationDriver::destroyResources() {
    state_.agents = {};
    gridBuildPipeline_ = {};
    gridClearPipeline_ = {};
    stepPipeline_ = {};
    descriptorAllocator_.reset();
    gridBuildDescriptorSetLayout_.reset();
    gridClearDescriptorSetLayout_.reset();
    stepDescriptorSetLayout_.reset();
    gridBuildDescriptorSets_ = {};
    gridClearDescriptorSet_ = VK_NULL_HANDLE;
    stepDescriptorSets_ = {};
    gridNext_.reset();
    gridHeads_.reset();
    stepParameterBuffer_.reset();
    genomeBuffer_.reset();
    agentBuffers_.reset();
    device_ = VK_NULL_HANDLE;
    physicalDevice_ = VK_NULL_HANDLE;
}

std::vector<AgentState> SimulationDriver::makeInitialAgents() const {
    const std::size_t genomeCount = evolution_.population().size();
    std::vector<AgentState> result(genomeCount * config_.trialsPerGenome);
    constexpr float goldenAngle = 2.39996323F;
    const SimulationStep initialSettings = resolveStepSettings(state_.physics, 0, 0);
    for (std::size_t genome = 0; genome < genomeCount; ++genome) {
        const std::size_t group = genome / state_.worlds.agentsPerWorld;
        const std::size_t firstGenome = group * state_.worlds.agentsPerWorld;
        const std::size_t agentsInGroup =
            std::min<std::size_t>(state_.worlds.agentsPerWorld, genomeCount - firstGenome);
        const std::size_t agentInGroup = genome - firstGenome;
        const float normalizedRadius = std::sqrt((static_cast<float>(agentInGroup) + 0.5F) /
                                                 static_cast<float>(agentsInGroup));
        for (std::size_t trial = 0; trial < config_.trialsPerGenome; ++trial) {
            const std::size_t index = genome * config_.trialsPerGenome + trial;
            const float positionAngle =
                goldenAngle * static_cast<float>(agentInGroup) + static_cast<float>(trial) * 0.43F;
            const float spawnRadius = normalizedRadius * state_.physics.worldRadius * 0.70F;
            const float heading =
                std::fmod(positionAngle * 1.73F + 0.37F, 2.0F * std::numbers::pi_v<float>);
            AgentState agent{};
            agent.pose = {std::cos(positionAngle) * spawnRadius,
                          std::sin(positionAngle) * spawnRadius, heading, 0.022F};
            agent.motion.w = 1.0F;
            agent.signal = {0.15F, 0.45F, 0.85F, 0.0F};
            const Float4 target = stationaryBeaconPosition(static_cast<std::uint32_t>(trial),
                                                           state_.physics.worldRadius);
            agent.target = {target.x, target.y, static_cast<float>(trial), 0.0F};
            const float distance = nearestBeaconDistance(agent, initialSettings);
            agent.metrics = {distance, distance, 0.0F, 0.0F};
            agent.penalties.w = static_cast<float>(
                logicalWorldForAgent(static_cast<std::uint32_t>(index),
                                     state_.worlds.agentsPerWorld, config_.trialsPerGenome));
            result[index] = agent;
        }
    }
    return result;
}

void SimulationDriver::uploadPopulation() {
    std::vector<float> flattened;
    flattened.reserve(evolution_.population().size() * neuro::Topology::weightCount);
    for (const Genome& genome : evolution_.population()) {
        flattened.insert(flattened.end(), genome.weights.begin(), genome.weights.end());
    }
    genomeBuffer_.write(flattened.data(), flattened.size() * sizeof(float));
    agentBuffers_.read().write(agents_.data(), agents_.size() * sizeof(AgentState));
    agentBuffers_.write().write(agents_.data(), agents_.size() * sizeof(AgentState));
    hostUploadPending_ = true;
}

void SimulationDriver::resetGeneration() {
    // The beacon motion seed is a property of the generation, so publishing it
    // here keeps simulation and visualization on the same random world.
    state_.physics.beaconMotionSeed = static_cast<std::uint32_t>(evolution_.generation());
    agents_ = makeInitialAgents();
    uploadPopulation();
    state_.statistics.step = 0;
    state_.statistics.generation = evolution_.generation();
    state_.agents.generation = evolution_.generation();
    state_.agents.currentIndex = agentBuffers_.readIndex();
}

void SimulationDriver::restart() {
    evolution_.reset();
    state_.statistics = {};
    state_.history.bestFitness.clear();
    state_.history.medianFitness.clear();
    state_.history.meanFitness.clear();
    state_.history.arrivalRatio.clear();
    refreshGridForWorldSize();
    resetGeneration();
}

void SimulationDriver::loadPopulation(const std::span<const Genome> genomes,
                                      const std::uint64_t generation) {
    evolution_.setPopulation(genomes, generation);
    state_.statistics.generation = generation;
    resetGeneration();
}

bool SimulationDriver::generationComplete() const {
    return state_.statistics.step >= state_.controls.stepsPerGeneration;
}

GenerationSummary SimulationDriver::finishGeneration() {
    agentBuffers_.read().read(agents_.data(), agents_.size() * sizeof(AgentState));
    const ScenarioDefinition& scenario = scenarioDefinition(state_.physics.beaconScenario);
    std::vector<float> fitness(evolution_.population().size());
    std::size_t completedPhases = 0;
    for (std::size_t genome = 0; genome < fitness.size(); ++genome) {
        for (std::size_t trial = 0; trial < config_.trialsPerGenome; ++trial) {
            const AgentState& agent = agents_[genome * config_.trialsPerGenome + trial];
            fitness[genome] += scenario.fitness(agent);
            completedPhases += state_.physics.beaconScenario == BeaconScenario::ForageHome
                                   ? static_cast<std::size_t>(completedForageCycles(agent) > 0)
                                   : completedBeaconPhases(agent);
        }
        fitness[genome] /= static_cast<float>(config_.trialsPerGenome);
    }
    const GenerationSummary summary = evolution_.evolve(fitness);
    state_.statistics.bestFitness = summary.bestFitness;
    state_.statistics.meanFitness = summary.meanFitness;
    state_.statistics.medianFitness = summary.medianFitness;
    const std::size_t phasesPerAgent =
        state_.physics.beaconScenario == BeaconScenario::AlternatingDiagonals ? 2U : 1U;
    state_.statistics.arrivalRatio =
        static_cast<float>(completedPhases) / static_cast<float>(agents_.size() * phasesPerAgent);
    const auto appendHistory = [&](std::vector<float>& history, const float value) {
        history.push_back(value);
        if (history.size() > state_.history.maximumSamples) {
            history.erase(history.begin(),
                          history.begin() + static_cast<std::ptrdiff_t>(
                                                history.size() - state_.history.maximumSamples));
        }
    };
    appendHistory(state_.history.bestFitness, summary.bestFitness);
    appendHistory(state_.history.medianFitness, summary.medianFitness);
    appendHistory(state_.history.meanFitness, summary.meanFitness);
    appendHistory(state_.history.arrivalRatio, state_.statistics.arrivalRatio);
    resetGeneration();
    return summary;
}

void SimulationDriver::updateGridDimensions() {
    gridWidth_ = static_cast<std::uint32_t>(
        std::ceil((state_.physics.worldRadius * 2.0F) / config_.gridCellSize));
    gridCellsPerWorld_ = gridWidth_ * gridWidth_;
}

void SimulationDriver::updateWorldLayout() {
    const auto genomeCount = static_cast<std::uint32_t>(evolution_.population().size());
    state_.worlds.agentsPerWorld =
        clampAgentsPerWorld(genomeCount, state_.worlds.requestedAgentsPerWorld);
    state_.worlds.requestedAgentsPerWorld = state_.worlds.agentsPerWorld;
    state_.worlds.groupCount = worldGroupCount(genomeCount, state_.worlds.agentsPerWorld);
    state_.worlds.worldCount =
        logicalWorldCount(genomeCount, state_.worlds.agentsPerWorld, config_.trialsPerGenome);
    if (state_.worlds.worldCount == 0) {
        state_.worlds.selectedWorld = 0;
    } else {
        state_.worlds.selectedWorld =
            std::min(state_.worlds.selectedWorld, state_.worlds.worldCount - 1);
    }
}

void SimulationDriver::refreshGridForWorldSize() {
    updateWorldLayout();
    ensureGridCapacity();
    updateGridDimensions();
}

void SimulationDriver::ensureGridCapacity() {
    const auto maximumGridWidth = static_cast<std::uint32_t>(
        std::ceil((worldRadiusForSize(WorldSize::Large) * 2.0F) / config_.gridCellSize));
    const VkDeviceSize requiredBytes =
        sizeof(std::int32_t) * maximumGridWidth * maximumGridWidth * state_.worlds.worldCount;
    if (gridHeads_.size() >= requiredBytes) {
        return;
    }
    gridHeads_.reset();
    gridHeads_.create(physicalDevice_, device_,
                      {requiredBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});
    updateGridDescriptors();
}

void SimulationDriver::updateGridDescriptors() {
    if (gridClearDescriptorSet_ != VK_NULL_HANDLE) {
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                         gridHeads_.size())
            .update(device_, gridClearDescriptorSet_);
    }
    for (std::size_t index = 0; index < stepDescriptorSets_.size(); ++index) {
        if (stepDescriptorSets_[index] != VK_NULL_HANDLE) {
            DescriptorSetWriter{}
                .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                             gridHeads_.size())
                .update(device_, stepDescriptorSets_[index]);
        }
        if (gridBuildDescriptorSets_[index] != VK_NULL_HANDLE) {
            DescriptorSetWriter{}
                .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                             gridHeads_.size())
                .update(device_, gridBuildDescriptorSets_[index]);
        }
    }
}

GpuStepParameters SimulationDriver::stepParameters(const std::uint32_t generationStep) const {
    const ScenarioDefinition& scenario = scenarioDefinition(state_.physics.beaconScenario);
    const SimulationStep settings =
        resolveStepSettings(state_.physics, generationStep, state_.controls.stepsPerGeneration);
    return {settings.deltaTime,
            settings.worldRadius,
            settings.thrust,
            settings.turnAcceleration,
            settings.linearDrag,
            settings.angularDrag,
            settings.sensorFieldOfView,
            beaconArrivalRadius(settings),
            settings.maximumSpeed,
            settings.maximumAngularSpeed,
            settings.lightSensorRange,
            settings.lightExposure,
            settings.collisionRestitution,
            settings.collisionStiffness,
            config_.gridCellSize,
            settings.wallCollisionPenalty,
            state_.agents.agentCount,
            neuro::packBrainLayout(scenario.brain),
            config_.trialsPerGenome,
            static_cast<std::uint32_t>(settings.worldShape),
            gridWidth_,
            gridCellsPerWorld_,
            settings.agentCollisionsEnabled ? 1U : 0U,
            settings.agentLightEnabled ? 1U : 0U,
            static_cast<std::uint32_t>(settings.beaconScenario),
            settings.beaconPhase,
            settings.beaconPhaseChanged ? 1U : 0U,
            0U,
            scenario.gpuParameters(settings)};
}

std::uint32_t SimulationDriver::recordSteps(const VkCommandBuffer commands,
                                            const std::uint32_t maximumSteps) {
    const std::uint32_t remaining =
        state_.controls.stepsPerGeneration -
        std::min(state_.statistics.step, state_.controls.stepsPerGeneration);
    const std::uint32_t stepCount =
        std::min({maximumSteps, remaining, config_.maximumStepsPerBatch});
    if (stepCount == 0) {
        return 0;
    }

    for (std::uint32_t step = 0; step < stepCount; ++step) {
        stepParameterStaging_[step] = stepParameters(state_.statistics.step + step);
    }
    stepParameterBuffer_.write(stepParameterStaging_.data(), sizeof(GpuStepParameters) * stepCount);
    cmdBufferBarrier(commands, stepParameterBuffer_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                     VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    if (hostUploadPending_) {
        for (const VkBuffer buffer : state_.agents.buffers) {
            cmdBufferBarrier(commands, buffer, VK_PIPELINE_STAGE_2_HOST_BIT,
                             VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        cmdBufferBarrier(commands, genomeBuffer_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        hostUploadPending_ = false;
    }

    const GridBuildParameters gridParameters{state_.physics.worldRadius,
                                             config_.gridCellSize,
                                             state_.agents.agentCount,
                                             state_.worlds.worldCount,
                                             gridWidth_,
                                             gridCellsPerWorld_,
                                             0,
                                             0};
    const std::uint32_t totalGridCells = gridCellsPerWorld_ * state_.worlds.worldCount;
    const std::array<VkDeviceSize, 1> clearRanges{gridHeads_.size()};
    const DispatchSize gridGroups = checkedDispatchSize(
        physicalDevice_, {{totalGridCells, 1, 1}, {64, 1, 1}, sizeof(totalGridCells), clearRanges});
    const std::array<VkDeviceSize, 3> buildRanges{agentBuffers_.read().size(), gridHeads_.size(),
                                                  gridNext_.size()};
    const DispatchSize buildGroups = checkedDispatchSize(
        physicalDevice_,
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(gridParameters), buildRanges});
    const std::array<VkDeviceSize, 6> stepRanges{agentBuffers_.read().size(),
                                                 agentBuffers_.write().size(),
                                                 genomeBuffer_.size(),
                                                 gridHeads_.size(),
                                                 gridNext_.size(),
                                                 stepParameterBuffer_.size()};
    const DispatchSize stepGroups = checkedDispatchSize(
        physicalDevice_,
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(std::uint32_t), stepRanges});

    for (std::uint32_t step = 0; step < stepCount; ++step) {
        cmdBufferBarrier(commands, gridHeads_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        cmdBufferBarrier(commands, gridNext_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, gridClearPipeline_.pipeline());
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                gridClearPipeline_.layout(), 0, 1, &gridClearDescriptorSet_, 0,
                                nullptr);
        vkCmdPushConstants(commands, gridClearPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(totalGridCells), &totalGridCells);
        vkCmdDispatch(commands, gridGroups.x, 1, 1);
        cmdBufferBarrier(
            commands, gridHeads_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        const std::uint32_t readIndex = agentBuffers_.readIndex();
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, gridBuildPipeline_.pipeline());
        const VkDescriptorSet buildSet = gridBuildDescriptorSets_[readIndex];
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                gridBuildPipeline_.layout(), 0, 1, &buildSet, 0, nullptr);
        vkCmdPushConstants(commands, gridBuildPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(gridParameters), &gridParameters);
        vkCmdDispatch(commands, buildGroups.x, 1, 1);
        cmdComputeWriteToComputeRead(commands, gridHeads_.buffer());
        cmdComputeWriteToComputeRead(commands, gridNext_.buffer());

        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, stepPipeline_.pipeline());
        const VkDescriptorSet stepSet = stepDescriptorSets_[readIndex];
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, stepPipeline_.layout(), 0,
                                1, &stepSet, 0, nullptr);
        vkCmdPushConstants(commands, stepPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(step), &step);
        vkCmdDispatch(commands, stepGroups.x, 1, 1);
        cmdBufferBarrier(
            commands, agentBuffers_.read().buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        cmdComputeWriteToComputeRead(commands, agentBuffers_.write().buffer());
        agentBuffers_.swap();
        state_.agents.currentIndex = agentBuffers_.readIndex();
    }

    cmdBufferBarrier(commands, agentBuffers_.read().buffer(),
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_HOST_READ_BIT);
    state_.statistics.step += stepCount;
    return stepCount;
}

} // namespace vkexp
