#pragma once

#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/core/Module.hpp"
#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/SimulationState.hpp"

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
    [[nodiscard]] std::vector<AgentState> makeInitialAgents() const;

    SimulationState& state_;
    GeneticAlgorithm evolution_;
    ProfileMetricId metric_{invalidProfileMetric};
    BufferResource agentBuffer_;
    BufferResource genomeBuffer_;
    UniqueDescriptorSetLayout descriptorSetLayout_;
    DescriptorAllocator descriptorAllocator_;
    VkDescriptorSet descriptorSet_{};
    ComputePipeline pipeline_;
    std::vector<AgentState> agents_;
    bool finishPending_{};
    bool hostUploadPending_{};
};

} // namespace vkexp
