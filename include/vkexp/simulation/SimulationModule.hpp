#pragma once

#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/core/Module.hpp"
#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <array>
#include <vector>

namespace vkexp {

class Profiler;

class SimulationModule final : public Module {
public:
    SimulationModule(SimulationState& state, Profiler& profiler, EvolutionSettings evolution = {});

    void onAttach(AppContext& context) override;
    void onUpdate(AppContext& context, const FrameInfo& frame) override;
    void onRender(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

private:
    static constexpr std::uint32_t trialsPerGenome = 4;

    void createGpuResources(AppContext& context);
    void resetGeneration();
    void finishGeneration();
    void uploadPopulation();
    void updateWorldLayout();
    void ensureGridCapacity(AppContext& context);
    void updateGridDescriptors(VkDevice device);
    void updateGridDimensions();
    [[nodiscard]] GpuStepParameters stepParameters(std::uint32_t generationStep) const;
    [[nodiscard]] std::vector<AgentState> makeInitialAgents() const;

    SimulationState& state_;
    GeneticAlgorithm evolution_;
    ProfileMetricId metric_{invalidProfileMetric};
    static constexpr float gridCellSize = 0.12F;

    PingPongBuffer agentBuffers_;
    BufferResource genomeBuffer_;
    BufferResource gridHeads_;
    BufferResource gridNext_;
    UniqueDescriptorSetLayout stepDescriptorSetLayout_;
    UniqueDescriptorSetLayout gridClearDescriptorSetLayout_;
    UniqueDescriptorSetLayout gridBuildDescriptorSetLayout_;
    DescriptorAllocator descriptorAllocator_;
    std::array<VkDescriptorSet, 2> stepDescriptorSets_{};
    VkDescriptorSet gridClearDescriptorSet_{};
    std::array<VkDescriptorSet, 2> gridBuildDescriptorSets_{};
    ComputePipeline stepPipeline_;
    ComputePipeline gridClearPipeline_;
    ComputePipeline gridBuildPipeline_;
    std::vector<AgentState> agents_;
    std::uint32_t gridWidth_{};
    std::uint32_t gridCellsPerWorld_{};
    bool finishPending_{};
    bool hostUploadPending_{};
};

} // namespace vkexp
