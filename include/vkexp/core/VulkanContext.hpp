#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace vkexp {

class Window;
struct FrameSynchronizationTimings {
    double fenceWaitMs{};
    double acquireWaitMs{};
    double queueSubmitMs{};
    double presentMs{};
};

class VulkanContext {
public:
    explicit VulkanContext(Window& window, bool enableValidation = true);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    [[nodiscard]] bool beginFrame(FrameSynchronizationTimings* timings = nullptr);
    void endFrame(FrameSynchronizationTimings* timings = nullptr);
    void beginColorPass(VkAttachmentLoadOp loadOp, const VkClearColorValue& clearColor) const;
    void endColorPass() const;
    void waitIdle() const;

    [[nodiscard]] VkInstance instance() const { return instance_; }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    [[nodiscard]] VkDevice device() const { return device_; }
    [[nodiscard]] VkQueue graphicsQueue() const { return graphicsQueue_; }
    [[nodiscard]] std::uint32_t graphicsQueueFamily() const { return graphicsQueueFamily_; }
    [[nodiscard]] VkFormat colorFormat() const { return surfaceFormat_.format; }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] std::uint32_t imageCount() const {
        return static_cast<std::uint32_t>(swapchainImages_.size());
    }
    [[nodiscard]] VkCommandBuffer commandBuffer() const { return commandBuffers_[currentFrame_]; }

private:
    // The experiment images are shared by graphics, compute, and ImGui.
    // One in-flight frame keeps their ownership deterministic.
    static constexpr std::size_t framesInFlight = 1;

    struct QueueFamilies {
        std::optional<std::uint32_t> graphics;
        std::optional<std::uint32_t> present;
        [[nodiscard]] bool complete() const { return graphics.has_value() && present.has_value(); }
    };

    [[nodiscard]] QueueFamilies findQueueFamilies(VkPhysicalDevice device) const;
    [[nodiscard]] bool deviceSuitable(VkPhysicalDevice device) const;
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData);
    void cleanup();
    void createInstance(bool enableValidation);
    void createDebugMessenger();
    void destroyDebugMessenger();
    void createSurface();
    void selectPhysicalDevice();
    void createDevice();
    void createSwapchain();
    void destroySwapchain();
    void recreateSwapchain();
    void createCommands();
    void createSyncObjects();
    void transitionCurrentImage(VkImageLayout oldLayout, VkImageLayout newLayout) const;

    Window& window_;
    bool validationEnabled_{};
    bool debugUtilsEnabled_{};
    VkInstance instance_{};
    VkDebugUtilsMessengerEXT debugMessenger_{};
    VkSurfaceKHR surface_{};
    VkPhysicalDevice physicalDevice_{};
    VkDevice device_{};
    std::uint32_t graphicsQueueFamily_{};
    std::uint32_t presentQueueFamily_{};
    VkQueue graphicsQueue_{};
    VkQueue presentQueue_{};
    VkSwapchainKHR swapchain_{};
    VkSurfaceFormatKHR surfaceFormat_{};
    VkExtent2D extent_{};
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkImageLayout> imageLayouts_;
    std::vector<VkFence> imagesInFlight_;
    VkCommandPool commandPool_{};
    std::array<VkCommandBuffer, framesInFlight> commandBuffers_{};
    std::array<VkSemaphore, framesInFlight> imageAvailable_{};
    std::array<VkSemaphore, framesInFlight> renderFinished_{};
    std::array<VkFence, framesInFlight> inFlight_{};
    std::uint32_t currentImage_{};
    std::size_t currentFrame_{};
    bool frameActive_{};
};

} // namespace vkexp
