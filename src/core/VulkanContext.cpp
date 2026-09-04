#include "vkexp/core/VulkanContext.hpp"

#include "vkexp/core/Window.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>

namespace vkexp {
namespace {

constexpr std::array validationLayers{"VK_LAYER_KHRONOS_validation"};
constexpr std::array deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

void check(const VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string{operation} + " failed with VkResult " +
                                 std::to_string(result));
    }
}

bool validationAvailable() {
    std::uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    return std::ranges::any_of(layers, [](const VkLayerProperties& layer) {
        return std::strcmp(layer.layerName, validationLayers.front()) == 0;
    });
}

bool instanceExtensionAvailable(const char* wanted) {
    std::uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
    return std::ranges::any_of(extensions, [wanted](const VkExtensionProperties& extension) {
        return std::strcmp(extension.extensionName, wanted) == 0;
    });
}

} // namespace

VulkanContext::VulkanContext(Window& window, const bool enableValidation) : window_(window) {
    try {
        createInstance(enableValidation);
        createSurface();
        selectPhysicalDevice();
        createDevice();
        createSwapchain();
        createCommands();
        createSyncObjects();
    } catch (...) {
        cleanup();
        throw;
    }
}

VulkanContext::~VulkanContext() { cleanup(); }

void VulkanContext::cleanup() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        for (std::size_t i = 0; i < framesInFlight; ++i) {
            vkDestroyFence(device_, inFlight_[i], nullptr);
            vkDestroySemaphore(device_, renderFinished_[i], nullptr);
            vkDestroySemaphore(device_, imageAvailable_[i], nullptr);
        }
        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }
        destroySwapchain();
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    destroyDebugMessenger();
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
    instance_ = VK_NULL_HANDLE;
}

void VulkanContext::createInstance(const bool enableValidation) {
    validationEnabled_ = enableValidation && validationAvailable();

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "Vulkan Experiment Framework";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "vkexp";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr) {
        throw std::runtime_error("GLFW did not provide Vulkan instance extensions");
    }
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    debugUtilsEnabled_ =
        validationEnabled_ && instanceExtensionAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debugUtilsEnabled_) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    VkDebugUtilsMessengerCreateInfoEXT debugInfo{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    if (validationEnabled_) {
        createInfo.enabledLayerCount = static_cast<std::uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    if (debugUtilsEnabled_) {
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback = debugCallback;
        createInfo.pNext = &debugInfo;
    }
    check(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");
    createDebugMessenger();
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debugCallback(
    const VkDebugUtilsMessageSeverityFlagBitsEXT, const VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void*) {
    std::cerr << "[Vulkan validation] "
              << (callbackData != nullptr ? callbackData->pMessage : "Unknown message") << '\n';
    return VK_FALSE;
}

void VulkanContext::createDebugMessenger() {
    if (!debugUtilsEnabled_) {
        return;
    }
    const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (create == nullptr) {
        throw std::runtime_error("VK_EXT_debug_utils is enabled but unavailable");
    }
    VkDebugUtilsMessengerCreateInfoEXT info{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
    check(create(instance_, &info, nullptr, &debugMessenger_), "vkCreateDebugUtilsMessengerEXT");
}

void VulkanContext::destroyDebugMessenger() {
    if (debugMessenger_ == VK_NULL_HANDLE || instance_ == VK_NULL_HANDLE) {
        return;
    }
    const auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy != nullptr) {
        destroy(instance_, debugMessenger_, nullptr);
    }
    debugMessenger_ = VK_NULL_HANDLE;
}

void VulkanContext::createSurface() {
    check(glfwCreateWindowSurface(instance_, window_.handle(), nullptr, &surface_),
          "glfwCreateWindowSurface");
}

VulkanContext::QueueFamilies VulkanContext::findQueueFamilies(const VkPhysicalDevice device) const {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());

    QueueFamilies result;
    for (std::uint32_t index = 0; index < count; ++index) {
        if ((properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U &&
            (properties[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U) {
            result.graphics = index;
        }
        VkBool32 presentSupported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface_, &presentSupported);
        if (presentSupported == VK_TRUE) {
            result.present = index;
        }
        if (result.complete()) {
            break;
        }
    }
    return result;
}

bool VulkanContext::deviceSuitable(const VkPhysicalDevice device) const {
    const QueueFamilies queues = findQueueFamilies(device);
    if (!queues.complete()) {
        return false;
    }

    std::uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());
    for (const char* wanted : deviceExtensions) {
        if (!std::ranges::any_of(extensions, [wanted](const VkExtensionProperties& extension) {
                return std::strcmp(extension.extensionName, wanted) == 0;
            })) {
            return false;
        }
    }

    std::uint32_t formatCount = 0;
    std::uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, nullptr);

    VkPhysicalDeviceVulkan13Features vulkan13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &vulkan13;
    vkGetPhysicalDeviceFeatures2(device, &features);
    return formatCount > 0 && presentModeCount > 0 && vulkan13.dynamicRendering == VK_TRUE &&
           vulkan13.synchronization2 == VK_TRUE;
}

void VulkanContext::selectPhysicalDevice() {
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
    if (count == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
          "vkEnumeratePhysicalDevices");
    const auto found = std::ranges::find_if(
        devices, [this](const VkPhysicalDevice device) { return deviceSuitable(device); });
    if (found == devices.end()) {
        throw std::runtime_error(
            "No Vulkan 1.3 device with graphics, compute, and presentation found");
    }
    physicalDevice_ = *found;
}

void VulkanContext::createDevice() {
    const QueueFamilies families = findQueueFamilies(physicalDevice_);
    graphicsQueueFamily_ = *families.graphics;
    presentQueueFamily_ = *families.present;

    const std::set uniqueFamilies{graphicsQueueFamily_, presentQueueFamily_};
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    constexpr float priority = 1.0F;
    for (const std::uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        queueInfos.push_back(queueInfo);
    }

    VkPhysicalDeviceVulkan13Features vulkan13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    vulkan13.dynamicRendering = VK_TRUE;
    vulkan13.synchronization2 = VK_TRUE;

    VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.pNext = &vulkan13;
    createInfo.queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    check(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, graphicsQueueFamily_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, presentQueueFamily_, 0, &presentQueue_);
}

void VulkanContext::createSwapchain() {
    VkSurfaceCapabilitiesKHR capabilities{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &capabilities),
          "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    std::uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, formats.data());
    surfaceFormat_ = formats.front();
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surfaceFormat_ = format;
            break;
        }
    }

    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        extent_ = capabilities.currentExtent;
    } else {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_.handle(), &width, &height);
        extent_.width =
            std::clamp(static_cast<std::uint32_t>(width), capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width);
        extent_.height =
            std::clamp(static_cast<std::uint32_t>(height), capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height);
    }

    std::uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    }

    VkSwapchainCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat_.format;
    createInfo.imageColorSpace = surfaceFormat_.colorSpace;
    createInfo.imageExtent = extent_;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    const std::array queueFamilies{graphicsQueueFamily_, presentQueueFamily_};
    if (graphicsQueueFamily_ != presentQueueFamily_) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = static_cast<std::uint32_t>(queueFamilies.size());
        createInfo.pQueueFamilyIndices = queueFamilies.data();
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    createInfo.clipped = VK_TRUE;
    check(vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_), "vkCreateSwapchainKHR");

    check(vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr),
          "vkGetSwapchainImagesKHR");
    swapchainImages_.resize(imageCount);
    check(vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data()),
          "vkGetSwapchainImagesKHR");
    imageLayouts_.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);
    imagesInFlight_.assign(imageCount, VK_NULL_HANDLE);

    swapchainImageViews_.resize(imageCount);
    for (std::size_t i = 0; i < swapchainImages_.size(); ++i) {
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = swapchainImages_[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = surfaceFormat_.format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        check(vkCreateImageView(device_, &viewInfo, nullptr, &swapchainImageViews_[i]),
              "vkCreateImageView");
    }
}

void VulkanContext::destroySwapchain() {
    for (const VkImageView view : swapchainImageViews_) {
        vkDestroyImageView(device_, view, nullptr);
    }
    swapchainImageViews_.clear();
    swapchainImages_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanContext::recreateSwapchain() {
    window_.waitForVisibleFramebuffer();
    waitIdle();
    destroySwapchain();
    createSwapchain();
}

void VulkanContext::createCommands() {
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphicsQueueFamily_;
    check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocateInfo.commandPool = commandPool_;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = static_cast<std::uint32_t>(commandBuffers_.size());
    check(vkAllocateCommandBuffers(device_, &allocateInfo, commandBuffers_.data()),
          "vkAllocateCommandBuffers");
}

void VulkanContext::createSyncObjects() {
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (std::size_t i = 0; i < framesInFlight; ++i) {
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailable_[i]),
              "vkCreateSemaphore");
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinished_[i]),
              "vkCreateSemaphore");
        check(vkCreateFence(device_, &fenceInfo, nullptr, &inFlight_[i]), "vkCreateFence");
    }
}

bool VulkanContext::beginFrame(FrameSynchronizationTimings* timings) {
    using Clock = std::chrono::steady_clock;
    FrameSynchronizationTimings ignoredTimings;
    FrameSynchronizationTimings& frameTimings = timings != nullptr ? *timings : ignoredTimings;
    frameTimings = {};

    auto started = Clock::now();
    const VkResult frameFence =
        vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX);
    frameTimings.fenceWaitMs +=
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    check(frameFence, "vkWaitForFences");
    started = Clock::now();
    const VkResult acquired =
        vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_[currentFrame_],
                              VK_NULL_HANDLE, &currentImage_);
    frameTimings.acquireWaitMs =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return false;
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
        check(acquired, "vkAcquireNextImageKHR");
    }
    if (imagesInFlight_[currentImage_] != VK_NULL_HANDLE) {
        started = Clock::now();
        const VkResult imageFence =
            vkWaitForFences(device_, 1, &imagesInFlight_[currentImage_], VK_TRUE, UINT64_MAX);
        frameTimings.fenceWaitMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        check(imageFence, "vkWaitForFences(image)");
    }
    imagesInFlight_[currentImage_] = inFlight_[currentFrame_];
    check(vkResetFences(device_, 1, &inFlight_[currentFrame_]), "vkResetFences");
    check(vkResetCommandBuffer(commandBuffer(), 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(commandBuffer(), &beginInfo), "vkBeginCommandBuffer");
    transitionCurrentImage(imageLayouts_[currentImage_], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    imageLayouts_[currentImage_] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    frameActive_ = true;
    return true;
}

void VulkanContext::transitionCurrentImage(const VkImageLayout oldLayout,
                                           const VkImageLayout newLayout) const {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                               ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                               : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barrier.srcAccessMask =
        oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstStageMask = newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                               ? VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT
                               : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barrier.dstAccessMask =
        newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            ? 0
            : VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapchainImages_[currentImage_];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer(), &dependency);
}

void VulkanContext::beginColorPass(const VkAttachmentLoadOp loadOp,
                                   const VkClearColorValue& clearColor) const {
    VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    attachment.imageView = swapchainImageViews_[currentImage_];
    attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.loadOp = loadOp;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.clearValue.color = clearColor;
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea.extent = extent_;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &attachment;
    vkCmdBeginRendering(commandBuffer(), &rendering);
}

void VulkanContext::endColorPass() const { vkCmdEndRendering(commandBuffer()); }

void VulkanContext::endFrame(FrameSynchronizationTimings* timings) {
    transitionCurrentImage(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    imageLayouts_[currentImage_] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    check(vkEndCommandBuffer(commandBuffer()), "vkEndCommandBuffer");

    VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitInfo.semaphore = imageAvailable_[currentFrame_];
    waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    commandInfo.commandBuffer = commandBuffer();
    VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signalInfo.semaphore = renderFinished_[currentFrame_];
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitInfo;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalInfo;
    using Clock = std::chrono::steady_clock;
    auto started = Clock::now();
    const VkResult submitted =
        vkQueueSubmit2(graphicsQueue_, 1, &submitInfo, inFlight_[currentFrame_]);
    FrameSynchronizationTimings ignoredTimings;
    FrameSynchronizationTimings& frameTimings = timings != nullptr ? *timings : ignoredTimings;
    frameTimings.queueSubmitMs =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    check(submitted, "vkQueueSubmit2");

    VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished_[currentFrame_];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_;
    presentInfo.pImageIndices = &currentImage_;
    started = Clock::now();
    const VkResult presented = vkQueuePresentKHR(presentQueue_, &presentInfo);
    frameTimings.presentMs =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    frameActive_ = false;
    currentFrame_ = (currentFrame_ + 1) % framesInFlight;
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
        recreateSwapchain();
    } else {
        check(presented, "vkQueuePresentKHR");
    }
}

void VulkanContext::waitIdle() const {
    if (device_ != VK_NULL_HANDLE) {
        check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
    }
}

} // namespace vkexp
