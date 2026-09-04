#include "vkexp/compute/ComputeModule.hpp"

#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/demo/DemoState.hpp"
#include "vkexp/profiling/Profiler.hpp"

#include <array>
#include <stdexcept>

namespace vkexp {

ComputeModule::ComputeModule(DemoState& state, Profiler& profiler)
    : state_(state), metric_(profiler.registerMetric("Compute blur")) {}

void ComputeModule::onAttach(AppContext& context) {
    const VkDevice device = context.vulkan.device();

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1] = bindings[0];
    bindings[1].binding = 1;
    VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorLayoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptorLayoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr,
                                    descriptorSetLayout_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create blur descriptor set layout");
    }

    const VkDescriptorSetLayout descriptorSetLayout = descriptorSetLayout_.get();
    descriptorAllocator_.create(
        device, DescriptorAllocatorConfig{1, {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}}});
    descriptorSet_ = descriptorAllocator_.allocate(descriptorSetLayout);
    pipeline_ = ComputePipelineBuilder{context.vulkan.physicalDevice(), device}
                    .shader(VKEXP_SHADER_DIR "/experiment.comp.spv")
                    .addDescriptorSetLayout(descriptorSetLayout)
                    .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(int))
                    .build();
    createOutput(context);
    updateDescriptors(context);
    sourceGeneration_ = state_.viewport.generation;
}

void ComputeModule::createOutput(AppContext& context) {
    const VkExtent2D extent = state_.viewport.extent;
    output_.create(context.vulkan.physicalDevice(), context.vulkan.device(),
                   ImageResourceConfig{
                       extent,
                       outputFormat,
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                   });

    outputLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    state_.blur.imageView = output_.view();
    state_.blur.sampler = output_.sampler();
    state_.blur.extent = extent;
    state_.blur.ready = false;
    ++state_.blur.generation;
}

void ComputeModule::destroyOutput() {
    state_.blur.imageView = VK_NULL_HANDLE;
    state_.blur.sampler = VK_NULL_HANDLE;
    state_.blur.ready = false;
    output_.reset();
    outputLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void ComputeModule::updateDescriptors(AppContext& context) {
    DescriptorSetWriter{}
        .writeImage(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, state_.viewport.imageView,
                    VK_IMAGE_LAYOUT_GENERAL)
        .writeImage(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, output_.view(), VK_IMAGE_LAYOUT_GENERAL)
        .update(context.vulkan.device(), descriptorSet_);
}

void ComputeModule::onUpdate(AppContext& context, const FrameInfo&) {
    if (sourceGeneration_ == state_.viewport.generation) {
        return;
    }
    context.vulkan.waitIdle();
    destroyOutput();
    createOutput(context);
    updateDescriptors(context);
    sourceGeneration_ = state_.viewport.generation;
}

void ComputeModule::onRender(AppContext& context, const FrameInfo&) {
    if (!state_.preset.computeEnabled) {
        state_.blur.requested = false;
        return;
    }
    if (!state_.blur.requested) {
        return;
    }

    auto cpuScope = context.profiler.cpu().scope(metric_);
    auto gpuScope = context.profiler.gpu().scope(context.vulkan.commandBuffer(), metric_);
    const VkCommandBuffer commands = context.vulkan.commandBuffer();
    cmdImageBarrier(commands, state_.viewport.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    cmdImageBarrier(
        commands, output_.image(), outputLayout_, VK_IMAGE_LAYOUT_GENERAL,
        outputLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                                   : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        outputLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.pipeline());
    const VkPipelineLayout pipelineLayout = pipeline_.layout();
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                            &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int),
                       &state_.blur.radius);
    const DispatchSize groups = checkedDispatchSize(
        context.vulkan.physicalDevice(),
        {{state_.blur.extent.width, state_.blur.extent.height, 1}, {8, 8, 1}, sizeof(int)});
    vkCmdDispatch(commands, groups.x, groups.y, groups.z);

    cmdImageBarrier(commands, state_.viewport.image, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    cmdImageBarrier(commands, output_.image(), VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    outputLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    state_.blur.requested = false;
    state_.blur.ready = true;
}

void ComputeModule::onDetach(AppContext&) {
    destroyOutput();
    pipeline_ = {};
    descriptorAllocator_.reset();
    descriptorSetLayout_.reset();
    descriptorSet_ = VK_NULL_HANDLE;
}

} // namespace vkexp
