#include "vkexp/simulation/SimulationModule.hpp"

#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/profiling/Profiler.hpp"
#include "vkexp/simulation/Beacons.hpp"
#include "vkexp/simulation/CpuSimulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
    std::uint32_t trialsPerGenome{};
    std::uint32_t gridWidth{};
    std::uint32_t gridCellsPerTrial{};
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

SimulationModule::SimulationModule(SimulationState& state, Profiler& profiler,
                                   EvolutionSettings evolution)
    : state_(state), evolution_(evolution), metric_(profiler.registerMetric("Agent simulation")) {
    state_.evolution = evolution_.settings();
}

void SimulationModule::onAttach(AppContext& context) {
    createGpuResources(context);
    resetGeneration();
}

void SimulationModule::createGpuResources(AppContext& context) {
    const VkDevice device = context.vulkan.device();
    const VkPhysicalDevice physicalDevice = context.vulkan.physicalDevice();
    const std::uint32_t agentCount =
        static_cast<std::uint32_t>(evolution_.population().size()) * trialsPerGenome;
    const VkDeviceSize agentBytes = sizeof(AgentState) * agentCount;
    const VkDeviceSize genomeBytes =
        sizeof(float) * neuro::Topology::weightCount * evolution_.population().size();
    agentBuffers_.create(physicalDevice, device,
                         {agentBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    genomeBuffer_.create(physicalDevice, device,
                         {genomeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});

    updateGridDimensions();
    const auto maximumGridWidth = static_cast<std::uint32_t>(
        std::ceil((worldRadiusForSize(WorldSize::Large) * 2.0F) / gridCellSize));
    const std::uint32_t maximumGridCellsPerTrial = maximumGridWidth * maximumGridWidth;
    const VkDeviceSize gridHeadBytes =
        sizeof(std::int32_t) * maximumGridCellsPerTrial * trialsPerGenome;
    const VkDeviceSize gridLinkBytes = sizeof(std::int32_t) * agentCount;
    gridHeads_.create(physicalDevice, device, {gridHeadBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});
    gridNext_.create(physicalDevice, device, {gridLinkBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});

    createStorageLayout(device, 5, stepDescriptorSetLayout_);
    createStorageLayout(device, 1, gridClearDescriptorSetLayout_);
    createStorageLayout(device, 3, gridBuildDescriptorSetLayout_);
    descriptorAllocator_.create(device, {5, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 17}}});

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
            .update(device, stepDescriptorSets_[readIndex]);

        gridBuildDescriptorSets_[readIndex] =
            descriptorAllocator_.allocate(gridBuildDescriptorSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, readBuffer, 0, agentBytes)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                         gridHeads_.size())
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridNext_.buffer(), 0,
                         gridNext_.size())
            .update(device, gridBuildDescriptorSets_[readIndex]);
    }
    gridClearDescriptorSet_ = descriptorAllocator_.allocate(gridClearDescriptorSetLayout_.get());
    DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                     gridHeads_.size())
        .update(device, gridClearDescriptorSet_);

    stepPipeline_ =
        ComputePipelineBuilder{physicalDevice, device}
            .shader(VKEXP_SHADER_DIR "/agent_step.comp.spv")
            .addDescriptorSetLayout(stepDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(GpuStepParameters))
            .build();
    gridClearPipeline_ =
        ComputePipelineBuilder{physicalDevice, device}
            .shader(VKEXP_SHADER_DIR "/agent_grid_clear.comp.spv")
            .addDescriptorSetLayout(gridClearDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
            .build();
    gridBuildPipeline_ =
        ComputePipelineBuilder{physicalDevice, device}
            .shader(VKEXP_SHADER_DIR "/agent_grid_build.comp.spv")
            .addDescriptorSetLayout(gridBuildDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(GridBuildParameters))
            .build();

    state_.agents = {{agentBuffers_.read().buffer(), agentBuffers_.write().buffer()},
                     agentBytes,
                     agentBuffers_.readIndex(),
                     agentCount,
                     static_cast<std::uint32_t>(evolution_.population().size()),
                     trialsPerGenome,
                     0};
}

std::vector<AgentState> SimulationModule::makeInitialAgents() const {
    const std::size_t genomeCount = evolution_.population().size();
    std::vector<AgentState> result(genomeCount * trialsPerGenome);
    constexpr float goldenAngle = 2.39996323F;
    SimulationStep initialSettings = state_.physics;
    initialSettings.beaconPhase = 0;
    initialSettings.beaconPhaseChanged = false;
    for (std::size_t genome = 0; genome < genomeCount; ++genome) {
        const float normalizedRadius =
            std::sqrt((static_cast<float>(genome) + 0.5F) / static_cast<float>(genomeCount));
        for (std::size_t trial = 0; trial < trialsPerGenome; ++trial) {
            const std::size_t index = genome * trialsPerGenome + trial;
            const float positionAngle =
                goldenAngle * static_cast<float>(genome) + static_cast<float>(trial) * 0.43F;
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
            result[index] = agent;
        }
    }
    return result;
}

void SimulationModule::uploadPopulation() {
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

void SimulationModule::resetGeneration() {
    agents_ = makeInitialAgents();
    uploadPopulation();
    state_.statistics.step = 0;
    state_.statistics.generation = evolution_.generation();
    state_.agents.generation = evolution_.generation();
    state_.agents.currentIndex = agentBuffers_.readIndex();
    finishPending_ = false;
}

void SimulationModule::finishGeneration() {
    agentBuffers_.read().read(agents_.data(), agents_.size() * sizeof(AgentState));
    std::vector<float> fitness(evolution_.population().size());
    std::size_t completedPhases = 0;
    for (std::size_t genome = 0; genome < fitness.size(); ++genome) {
        for (std::size_t trial = 0; trial < trialsPerGenome; ++trial) {
            const AgentState& agent = agents_[genome * trialsPerGenome + trial];
            fitness[genome] += agentFitness(agent);
            completedPhases += completedBeaconPhases(agent);
        }
        fitness[genome] /= static_cast<float>(trialsPerGenome);
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
}

void SimulationModule::updateGridDimensions() {
    gridWidth_ =
        static_cast<std::uint32_t>(std::ceil((state_.physics.worldRadius * 2.0F) / gridCellSize));
    gridCellsPerTrial_ = gridWidth_ * gridWidth_;
}

GpuStepParameters SimulationModule::stepParameters(const std::uint32_t generationStep) const {
    const std::uint32_t beaconPhase = beaconPhaseForStep(
        state_.physics.beaconScenario, generationStep, state_.controls.stepsPerGeneration);
    const bool phaseChanged =
        state_.physics.beaconScenario == BeaconScenario::AlternatingDiagonals &&
        generationStep == state_.controls.stepsPerGeneration / 2;
    return {state_.physics.deltaTime,
            state_.physics.worldRadius,
            state_.physics.thrust,
            state_.physics.turnAcceleration,
            state_.physics.linearDrag,
            state_.physics.angularDrag,
            state_.physics.sensorFieldOfView,
            state_.physics.arrivalRadius,
            state_.physics.maximumSpeed,
            state_.physics.maximumAngularSpeed,
            state_.physics.lightSensorRange,
            state_.physics.lightExposure,
            state_.physics.collisionRestitution,
            state_.physics.collisionStiffness,
            gridCellSize,
            0.0F,
            state_.agents.agentCount,
            static_cast<std::uint32_t>(neuro::Topology::weightCount),
            trialsPerGenome,
            static_cast<std::uint32_t>(state_.physics.worldShape),
            gridWidth_,
            gridCellsPerTrial_,
            state_.physics.agentCollisionsEnabled ? 1U : 0U,
            state_.physics.agentLightEnabled ? 1U : 0U,
            static_cast<std::uint32_t>(state_.physics.beaconScenario),
            beaconPhase,
            phaseChanged ? 1U : 0U,
            0U};
}

void SimulationModule::onUpdate(AppContext& context, const FrameInfo&) {
    if (state_.controls.resetRequested) {
        context.vulkan.waitIdle();
        evolution_.reset();
        state_.statistics = {};
        state_.history.bestFitness.clear();
        state_.history.medianFitness.clear();
        state_.history.meanFitness.clear();
        state_.history.arrivalRatio.clear();
        updateGridDimensions();
        resetGeneration();
        state_.controls.resetRequested = false;
    } else if (finishPending_) {
        context.vulkan.waitIdle();
        finishGeneration();
    }
}

void SimulationModule::onRender(AppContext& context, const FrameInfo&) {
    if (state_.controls.paused || finishPending_) {
        return;
    }
    auto cpuScope = context.profiler.cpu().scope(metric_);
    auto gpuScope = context.profiler.gpu().scope(context.vulkan.commandBuffer(), metric_);
    const std::uint32_t remaining =
        state_.controls.stepsPerGeneration -
        std::min(state_.statistics.step, state_.controls.stepsPerGeneration);
    const std::uint32_t stepCount = std::min(state_.controls.stepsPerFrame, remaining);
    if (stepCount == 0) {
        finishPending_ = true;
        return;
    }

    const VkCommandBuffer commands = context.vulkan.commandBuffer();
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
                                             gridCellSize,
                                             state_.agents.agentCount,
                                             trialsPerGenome,
                                             gridWidth_,
                                             gridCellsPerTrial_,
                                             0,
                                             0};
    const std::uint32_t totalGridCells = gridCellsPerTrial_ * trialsPerGenome;
    const std::array<VkDeviceSize, 1> clearRanges{gridHeads_.size()};
    const DispatchSize gridGroups = checkedDispatchSize(
        context.vulkan.physicalDevice(),
        {{totalGridCells, 1, 1}, {64, 1, 1}, sizeof(totalGridCells), clearRanges});
    const std::array<VkDeviceSize, 3> buildRanges{agentBuffers_.read().size(), gridHeads_.size(),
                                                  gridNext_.size()};
    const DispatchSize buildGroups = checkedDispatchSize(
        context.vulkan.physicalDevice(),
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(gridParameters), buildRanges});
    const std::array<VkDeviceSize, 5> stepRanges{agentBuffers_.read().size(),
                                                 agentBuffers_.write().size(), genomeBuffer_.size(),
                                                 gridHeads_.size(), gridNext_.size()};
    const DispatchSize stepGroups = checkedDispatchSize(
        context.vulkan.physicalDevice(),
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(GpuStepParameters), stepRanges});

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
        const GpuStepParameters parameters = stepParameters(state_.statistics.step + step);
        vkCmdPushConstants(commands, stepPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(parameters), &parameters);
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
    if (state_.statistics.step >= state_.controls.stepsPerGeneration) {
        finishPending_ = true;
    }
}

void SimulationModule::onDetach(AppContext&) {
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
    genomeBuffer_.reset();
    agentBuffers_.reset();
}

} // namespace vkexp
