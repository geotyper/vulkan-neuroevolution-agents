#include "vkexp/graphics/GraphicsModule.hpp"

#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/demo/DemoState.hpp"
#include "vkexp/profiling/Profiler.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace vkexp {

GraphicsModule::GraphicsModule(DemoState& state, Profiler& profiler)
    : state_(state), metric_(profiler.registerMetric("Graphics")) {}

void GraphicsModule::onAttach(AppContext& context) {
    const VkDevice device = context.vulkan.device();
    const auto vertex = loadShaderModule(device, VKEXP_SHADER_DIR "/triangle.vert.spv");
    const auto fragment = loadShaderModule(device, VKEXP_SHADER_DIR "/triangle.frag.spv");

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, pipelineLayout_.put(device)) !=
        VK_SUCCESS) {
        throw std::runtime_error("Unable to create graphics pipeline layout");
    }

    const std::array stages{
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertex.get(),
                                        "main", nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragment.get(),
                                        "main", nullptr},
    };
    VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterizer{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0F;
    VkPipelineMultisampleStateCreateInfo multisampling{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blending.attachmentCount = 1;
    blending.pAttachments = &blendAttachment;
    constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();
    constexpr VkFormat colorFormat = targetFormat;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &colorFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &rendering;
    pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &blending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout_;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                  pipeline_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create graphics pipeline");
    }
    createRenderTarget(context, state_.viewport.extent);
}

void GraphicsModule::createRenderTarget(AppContext& context, const VkExtent2D extent) {
    target_.create(context.vulkan.physicalDevice(), context.vulkan.device(),
                   ImageResourceConfig{
                       extent,
                       targetFormat,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_STORAGE_BIT,
                   });

    targetLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    state_.viewport.image = target_.image();
    state_.viewport.imageView = target_.view();
    state_.viewport.sampler = target_.sampler();
    state_.viewport.extent = extent;
    ++state_.viewport.generation;
}

void GraphicsModule::destroyRenderTarget() {
    state_.viewport.image = VK_NULL_HANDLE;
    state_.viewport.imageView = VK_NULL_HANDLE;
    state_.viewport.sampler = VK_NULL_HANDLE;
    target_.reset();
    targetLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void GraphicsModule::onUpdate(AppContext& context, const FrameInfo&) {
    auto cpuScope = context.profiler.cpu().scope(metric_);
    const VkExtent2D requested{
        std::clamp(state_.viewport.requestedWidth, 64U, 4096U),
        std::clamp(state_.viewport.requestedHeight, 64U, 4096U),
    };
    if (requested.width == state_.viewport.extent.width &&
        requested.height == state_.viewport.extent.height) {
        return;
    }
    context.vulkan.waitIdle();
    destroyRenderTarget();
    createRenderTarget(context, requested);
}

void GraphicsModule::onRender(AppContext& context, const FrameInfo&) {
    auto cpuScope = context.profiler.cpu().scope(metric_);
    auto gpuScope = context.profiler.gpu().scope(context.vulkan.commandBuffer(), metric_);
    const VkCommandBuffer commands = context.vulkan.commandBuffer();
    VkImageMemoryBarrier2 toAttachment{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    toAttachment.srcStageMask = targetLayout_ == VK_IMAGE_LAYOUT_UNDEFINED
                                    ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                    : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    toAttachment.srcAccessMask =
        targetLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    toAttachment.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toAttachment.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toAttachment.oldLayout = targetLayout_;
    toAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toAttachment.image = target_.image();
    toAttachment.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toAttachment.subresourceRange.levelCount = 1;
    toAttachment.subresourceRange.layerCount = 1;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &toAttachment;
    vkCmdPipelineBarrier2(commands, &dependency);

    const auto& color = state_.preset.clearColor;
    const VkClearColorValue clear{{color.r, color.g, color.b, color.a}};
    VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    attachment.imageView = target_.view();
    attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.clearValue.color = clear;
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea.extent = state_.viewport.extent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &attachment;
    vkCmdBeginRendering(commands, &rendering);
    if (state_.preset.graphicsEnabled) {
        const VkExtent2D extent = state_.viewport.extent;
        const VkViewport viewport{
            0.0F, 0.0F, static_cast<float>(extent.width), static_cast<float>(extent.height),
            0.0F, 1.0F};
        const VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(commands, 0, 1, &viewport);
        vkCmdSetScissor(commands, 0, 1, &scissor);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdDraw(commands, 3, 1, 0, 0);
    }
    vkCmdEndRendering(commands);

    VkImageMemoryBarrier2 toSampled{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    toSampled.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toSampled.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toSampled.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    toSampled.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    toSampled.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toSampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toSampled.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSampled.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSampled.image = target_.image();
    toSampled.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toSampled.subresourceRange.levelCount = 1;
    toSampled.subresourceRange.layerCount = 1;
    dependency.pImageMemoryBarriers = &toSampled;
    vkCmdPipelineBarrier2(commands, &dependency);
    targetLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void GraphicsModule::onDetach(AppContext&) {
    destroyRenderTarget();
    pipeline_.reset();
    pipelineLayout_.reset();
}

} // namespace vkexp
