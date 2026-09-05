#pragma once

#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/core/Module.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <array>

namespace vkexp {

class Profiler;

class AgentRenderer final : public Module {
public:
    AgentRenderer(SimulationState& state, Profiler& profiler);

    void onAttach(AppContext& context) override;
    void onUpdate(AppContext& context, const FrameInfo& frame) override;
    void onRender(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

private:
    static constexpr VkFormat targetFormat = VK_FORMAT_R8G8B8A8_UNORM;

    void createPipeline(AppContext& context);
    void createTarget(AppContext& context, VkExtent2D extent);
    void destroyTarget();
    // Changing the trail resolution reallocates the field, which invalidates the
    // handle written into these sets at attach time. framesInFlight is 1, so the
    // previous frame has finished by the time this runs and the sets are idle.
    void refreshTrailDescriptor(VkDevice device);
    void draw(VkCommandBuffer commands, float scaleX, float scaleY, float worldRadius,
              std::uint32_t mode, float opacity, std::uint32_t vertices,
              std::uint32_t instances) const;

    SimulationState& state_;
    ProfileMetricId metric_{invalidProfileMetric};
    UniqueDescriptorSetLayout descriptorSetLayout_;
    DescriptorAllocator descriptorAllocator_;
    std::array<VkDescriptorSet, 2> descriptorSets_{};
    UniquePipelineLayout pipelineLayout_;
    UniquePipeline pipeline_;
    VkBuffer boundTrailBuffer_{};
    ImageResource target_;
    VkImageLayout targetLayout_{VK_IMAGE_LAYOUT_UNDEFINED};
};

} // namespace vkexp
