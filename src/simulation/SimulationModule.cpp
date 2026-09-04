#include "vkexp/simulation/SimulationModule.hpp"

#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/profiling/Profiler.hpp"
#include "vkexp/simulation/CpuSimulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace vkexp {
namespace {

constexpr VkMemoryPropertyFlags hostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

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
    const std::uint32_t agentCount =
        static_cast<std::uint32_t>(evolution_.population().size()) * trialsPerGenome;
    const VkDeviceSize agentBytes = sizeof(AgentState) * agentCount;
    const VkDeviceSize genomeBytes =
        sizeof(float) * neuro::Topology::weightCount * evolution_.population().size();
    agentBuffer_.create(context.vulkan.physicalDevice(), device,
                        {agentBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    genomeBuffer_.create(context.vulkan.physicalDevice(), device,
                         {genomeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t binding = 0; binding < bindings.size(); ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[binding].descriptorCount = 1;
        bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                    descriptorSetLayout_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create agent simulation descriptor layout");
    }
    descriptorAllocator_.create(device, {1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}}});
    descriptorSet_ = descriptorAllocator_.allocate(descriptorSetLayout_.get());
    DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, agentBuffer_.buffer(), 0,
                     agentBuffer_.size())
        .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, genomeBuffer_.buffer(), 0,
                     genomeBuffer_.size())
        .update(device, descriptorSet_);
    pipeline_ = ComputePipelineBuilder{context.vulkan.physicalDevice(), device}
                    .shader(VKEXP_SHADER_DIR "/agent_step.comp.spv")
                    .addDescriptorSetLayout(descriptorSetLayout_.get())
                    .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(GpuStepParameters))
                    .build();

    state_.agents = {agentBuffer_.buffer(),
                     agentBuffer_.size(),
                     agentCount,
                     static_cast<std::uint32_t>(evolution_.population().size()),
                     trialsPerGenome,
                     0};
}

std::vector<AgentState> SimulationModule::makeInitialAgents() const {
    const std::size_t genomeCount = evolution_.population().size();
    std::vector<AgentState> result(genomeCount * trialsPerGenome);
    constexpr std::array<Float4, trialsPerGenome> targets{
        Float4{1.28F, 0.96F, 0.0F, 0.0F}, Float4{-1.40F, 0.70F, 0.0F, 0.0F},
        Float4{-0.96F, -1.32F, 0.0F, 0.0F}, Float4{1.36F, -0.92F, 0.0F, 0.0F}};
    for (std::size_t genome = 0; genome < genomeCount; ++genome) {
        for (std::size_t trial = 0; trial < trialsPerGenome; ++trial) {
            const std::size_t index = genome * trialsPerGenome + trial;
            const float hash =
                std::fmod(static_cast<float>(genome * 97 + trial * 31), 997.0F) / 997.0F;
            const float startAngle = hash * 2.0F * std::numbers::pi_v<float>;
            AgentState agent{};
            agent.pose = {std::cos(startAngle) * 0.045F, std::sin(startAngle) * 0.045F,
                          startAngle + static_cast<float>(trial) * 0.37F, 0.022F};
            agent.motion.w = 1.0F;
            agent.signal = {0.15F, 0.45F, 0.85F, 0.0F};
            agent.target = {targets[trial].x, targets[trial].y, static_cast<float>(trial),
                            static_cast<float>(genome)};
            const float dx = agent.target.x - agent.pose.x;
            const float dy = agent.target.y - agent.pose.y;
            const float distance = std::sqrt(dx * dx + dy * dy);
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
    agentBuffer_.write(agents_.data(), agents_.size() * sizeof(AgentState));
    hostUploadPending_ = true;
}

void SimulationModule::resetGeneration() {
    agents_ = makeInitialAgents();
    uploadPopulation();
    state_.statistics.step = 0;
    state_.statistics.generation = evolution_.generation();
    state_.agents.generation = evolution_.generation();
    finishPending_ = false;
}

void SimulationModule::finishGeneration() {
    agentBuffer_.read(agents_.data(), agents_.size() * sizeof(AgentState));
    std::vector<float> fitness(evolution_.population().size());
    std::size_t arrivals = 0;
    for (std::size_t genome = 0; genome < fitness.size(); ++genome) {
        for (std::size_t trial = 0; trial < trialsPerGenome; ++trial) {
            const AgentState& agent = agents_[genome * trialsPerGenome + trial];
            fitness[genome] += agentFitness(agent);
            arrivals += agent.metrics.w > 0.5F ? 1U : 0U;
        }
        fitness[genome] /= static_cast<float>(trialsPerGenome);
    }
    const GenerationSummary summary = evolution_.evolve(fitness);
    state_.statistics.bestFitness = summary.bestFitness;
    state_.statistics.meanFitness = summary.meanFitness;
    state_.statistics.medianFitness = summary.medianFitness;
    state_.statistics.arrivalRatio =
        static_cast<float>(arrivals) / static_cast<float>(agents_.size());
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

void SimulationModule::onUpdate(AppContext& context, const FrameInfo&) {
    if (state_.controls.resetRequested) {
        context.vulkan.waitIdle();
        evolution_.reset();
        state_.statistics = {};
        state_.history.bestFitness.clear();
        state_.history.medianFitness.clear();
        state_.history.meanFitness.clear();
        state_.history.arrivalRatio.clear();
        resetGeneration();
        state_.controls.resetRequested = false;
    } else if (finishPending_) {
        // Generation boundaries are intentionally synchronous in v1. Evolution
        // is infrequent and this keeps the CPU algorithm easy to inspect.
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
        cmdBufferBarrier(commands, agentBuffer_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        cmdBufferBarrier(commands, genomeBuffer_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        hostUploadPending_ = false;
    }
    const GpuStepParameters parameters{state_.physics.deltaTime,
                                       state_.physics.worldRadius,
                                       state_.physics.thrust,
                                       state_.physics.turnAcceleration,
                                       state_.physics.linearDrag,
                                       state_.physics.angularDrag,
                                       state_.physics.sensorFieldOfView,
                                       state_.physics.arrivalRadius,
                                       state_.physics.maximumSpeed,
                                       state_.physics.maximumAngularSpeed,
                                       0.0F,
                                       0.0F,
                                       state_.agents.agentCount,
                                       static_cast<std::uint32_t>(neuro::Topology::weightCount),
                                       trialsPerGenome,
                                       static_cast<std::uint32_t>(state_.physics.worldShape)};
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.pipeline());
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.layout(), 0, 1,
                            &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(commands, pipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(parameters), &parameters);
    const std::array<VkDeviceSize, 2> ranges{agentBuffer_.size(), genomeBuffer_.size()};
    const DispatchSize groups = checkedDispatchSize(
        context.vulkan.physicalDevice(),
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(parameters), ranges});
    for (std::uint32_t step = 0; step < stepCount; ++step) {
        vkCmdDispatch(commands, groups.x, 1, 1);
        if (step + 1 < stepCount) {
            cmdComputeWriteToComputeRead(commands, agentBuffer_.buffer());
        }
    }
    cmdBufferBarrier(commands, agentBuffer_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_HOST_READ_BIT);
    state_.statistics.step += stepCount;
    if (state_.statistics.step >= state_.controls.stepsPerGeneration) {
        finishPending_ = true;
    }
}

void SimulationModule::onDetach(AppContext&) {
    state_.agents = {};
    pipeline_ = {};
    descriptorAllocator_.reset();
    descriptorSetLayout_.reset();
    descriptorSet_ = VK_NULL_HANDLE;
    genomeBuffer_.reset();
    agentBuffer_.reset();
}

} // namespace vkexp
