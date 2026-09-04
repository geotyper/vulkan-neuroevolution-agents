#include "vkexp/ui/ImGuiModule.hpp"

#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/core/Window.hpp"
#include "vkexp/profiling/Profiler.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <array>
#include <stdexcept>

namespace vkexp {

ImGuiModule::ImGuiModule(Profiler& profiler) : metric_(profiler.registerMetric("ImGui")) {}

void ImGuiModule::onAttach(AppContext& context) {
    constexpr std::array poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 128},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 128},
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 128;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(context.vulkan.device(), &poolInfo, nullptr,
                               descriptorPool_.put(context.vulkan.device())) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create ImGui descriptor pool");
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    iniPath_ = VKEXP_IMGUI_INI;
    ImGui::GetIO().IniFilename = iniPath_.c_str();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    if (!ImGui_ImplGlfw_InitForVulkan(context.window.handle(), true)) {
        throw std::runtime_error("Unable to initialize the ImGui GLFW backend");
    }

    const VkFormat colorFormat = context.vulkan.colorFormat();
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = context.vulkan.instance();
    initInfo.PhysicalDevice = context.vulkan.physicalDevice();
    initInfo.Device = context.vulkan.device();
    initInfo.QueueFamily = context.vulkan.graphicsQueueFamily();
    initInfo.Queue = context.vulkan.graphicsQueue();
    initInfo.DescriptorPool = descriptorPool_;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = context.vulkan.imageCount();
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo = renderingInfo;
    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        throw std::runtime_error("Unable to initialize the ImGui Vulkan backend");
    }
}

void ImGuiModule::onFrameBegin(AppContext& context, const FrameInfo&) {
    auto cpuScope = context.profiler.cpu().scope(metric_);
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    frameOpen_ = true;
}

void ImGuiModule::onRender(AppContext& context, const FrameInfo&) {
    ImGui::Render();
    frameOpen_ = false;
    auto cpuScope = context.profiler.cpu().scope(metric_);
    auto gpuScope = context.profiler.gpu().scope(context.vulkan.commandBuffer(), metric_);
    constexpr VkClearColorValue background{{0.008F, 0.011F, 0.018F, 1.0F}};
    context.vulkan.beginColorPass(VK_ATTACHMENT_LOAD_OP_CLEAR, background);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), context.vulkan.commandBuffer());
    context.vulkan.endColorPass();
}

void ImGuiModule::onFrameEnd(AppContext&, const FrameInfo&) {
    if (frameOpen_) {
        ImGui::EndFrame();
        frameOpen_ = false;
    }
}

void ImGuiModule::onDetach(AppContext&) {
    ImGui::SaveIniSettingsToDisk(iniPath_.c_str());
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    descriptorPool_.reset();
}

VkDescriptorSet ImGuiModule::addTexture(const VkSampler sampler, const VkImageView imageView,
                                        const VkImageLayout layout) const {
    if (sampler == VK_NULL_HANDLE || imageView == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }
    return ImGui_ImplVulkan_AddTexture(sampler, imageView, layout);
}

void ImGuiModule::removeTexture(const VkDescriptorSet descriptor) const {
    if (descriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(descriptor);
    }
}

} // namespace vkexp
