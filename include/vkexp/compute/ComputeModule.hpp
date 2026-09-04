#pragma once

#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/core/Module.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"

namespace vkexp {

class DemoState;
class Profiler;

class ComputeModule final : public Module {
public:
    ComputeModule(DemoState& state, Profiler& profiler);

    void onAttach(AppContext& context) override;
    void onUpdate(AppContext& context, const FrameInfo& frame) override;
    void onRender(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

private:
    static constexpr VkFormat outputFormat = VK_FORMAT_R8G8B8A8_UNORM;

    void createOutput(AppContext& context);
    void destroyOutput();
    void updateDescriptors(AppContext& context);

    DemoState& state_;
    ProfileMetricId metric_{invalidProfileMetric};
    UniqueDescriptorSetLayout descriptorSetLayout_;
    DescriptorAllocator descriptorAllocator_;
    VkDescriptorSet descriptorSet_{};
    ComputePipeline pipeline_;
    ImageResource output_;
    VkImageLayout outputLayout_{VK_IMAGE_LAYOUT_UNDEFINED};
    std::uint64_t sourceGeneration_{};
};

} // namespace vkexp
