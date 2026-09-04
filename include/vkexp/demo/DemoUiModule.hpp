#pragma once

#include "vkexp/core/Module.hpp"
#include "vkexp/profiling/ProfilerPanel.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexp {

class DemoState;
class ImGuiModule;
class Profiler;

class DemoUiModule final : public Module {
public:
    DemoUiModule(DemoState& state, ImGuiModule& imgui, Profiler& profiler);

    void onAttach(AppContext& context) override;
    void onUpdate(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

private:
    void syncTextures();

    DemoState& state_;
    ImGuiModule& imgui_;
    ProfileMetricId metric_{invalidProfileMetric};
    VkDescriptorSet viewportDescriptor_{};
    VkDescriptorSet blurDescriptor_{};
    std::uint64_t viewportGeneration_{};
    std::uint64_t blurGeneration_{};
    ProfilerPanel profilerPanel_;
};

} // namespace vkexp
