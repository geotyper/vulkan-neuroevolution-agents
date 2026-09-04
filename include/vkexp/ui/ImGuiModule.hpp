#pragma once

#include "vkexp/core/Module.hpp"
#include "vkexp/core/VulkanResource.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"

#include <vulkan/vulkan.h>

#include <string>

namespace vkexp {

class Profiler;

class ImGuiModule final : public Module {
public:
    explicit ImGuiModule(Profiler& profiler);

    void onAttach(AppContext& context) override;
    void onFrameBegin(AppContext& context, const FrameInfo& frame) override;
    void onRender(AppContext& context, const FrameInfo& frame) override;
    void onFrameEnd(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

    [[nodiscard]] VkDescriptorSet addTexture(VkSampler sampler, VkImageView imageView,
                                             VkImageLayout layout) const;
    void removeTexture(VkDescriptorSet descriptor) const;

private:
    UniqueDescriptorPool descriptorPool_;
    ProfileMetricId metric_{invalidProfileMetric};
    std::string iniPath_;
    bool frameOpen_{};
};

} // namespace vkexp
