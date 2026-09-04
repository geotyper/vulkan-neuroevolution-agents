#pragma once

#include "vkexp/core/Module.hpp"
#include "vkexp/profiling/ProfilerPanel.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <vulkan/vulkan.h>

namespace vkexp {

class ImGuiModule;
class Profiler;

class SimulationUiModule final : public Module {
public:
    SimulationUiModule(SimulationState& state, ImGuiModule& imgui, Profiler& profiler);

    void onAttach(AppContext& context) override;
    void onUpdate(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

private:
    void syncTexture();

    SimulationState& state_;
    ImGuiModule& imgui_;
    ProfileMetricId metric_{invalidProfileMetric};
    VkDescriptorSet viewportDescriptor_{};
    std::uint64_t viewportGeneration_{};
    ProfilerPanel profilerPanel_;
};

} // namespace vkexp
