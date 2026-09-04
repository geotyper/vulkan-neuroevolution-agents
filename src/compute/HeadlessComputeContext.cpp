#include "vkexp/compute/HeadlessComputeContext.hpp"

#include <optional>
#include <vector>

namespace vkexp {
namespace {

std::optional<std::uint32_t> findComputeQueueFamily(const VkPhysicalDevice device,
                                                    const bool preferDedicated) {
    std::uint32_t count{};
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    std::optional<std::uint32_t> fallback;
    for (std::uint32_t index = 0; index < count; ++index) {
        if (families[index].queueCount == 0 ||
            (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
            continue;
        }
        if (!fallback.has_value()) {
            fallback = index;
        }
        const bool dedicated = (families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0;
        if (preferDedicated && dedicated) {
            return index;
        }
    }
    return fallback;
}

} // namespace

HeadlessComputeContext::HeadlessComputeContext(const HeadlessComputeConfig& config) {
    if (config.applicationName.empty() || config.apiVersion < VK_API_VERSION_1_3) {
        throw std::invalid_argument("Headless compute requires an application name and Vulkan 1.3");
    }
    try {
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = config.applicationName.c_str();
        application.apiVersion = config.apiVersion;
        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo = &application;
        if (vkCreateInstance(&instanceInfo, nullptr, &instance_) != VK_SUCCESS) {
            throw HeadlessComputeUnavailable("Vulkan 1.3 instance is unavailable");
        }

        std::uint32_t deviceCount{};
        if (vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr) != VK_SUCCESS ||
            deviceCount == 0) {
            throw HeadlessComputeUnavailable("No Vulkan physical device is available");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        if (vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data()) != VK_SUCCESS) {
            throw HeadlessComputeUnavailable("Unable to enumerate Vulkan physical devices");
        }

        for (const VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            VkPhysicalDeviceVulkan13Features features{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features2.pNext = &features;
            vkGetPhysicalDeviceFeatures2(candidate, &features2);
            if (properties.apiVersion < config.apiVersion || features.synchronization2 != VK_TRUE) {
                continue;
            }
            const auto family =
                findComputeQueueFamily(candidate, config.preferDedicatedComputeQueue);
            if (!family.has_value()) {
                continue;
            }
            physicalDevice_ = candidate;
            queueFamily_ = *family;
            deviceName_ = properties.deviceName;
            break;
        }
        if (physicalDevice_ == VK_NULL_HANDLE) {
            throw HeadlessComputeUnavailable(
                "No compatible Vulkan 1.3 compute queue with synchronization2 is available");
        }

        constexpr float queuePriority = 1.0F;
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = queueFamily_;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;
        VkPhysicalDeviceVulkan13Features features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        features.synchronization2 = VK_TRUE;
        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.pNext = &features;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        if (vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_) != VK_SUCCESS) {
            throw HeadlessComputeUnavailable("Unable to create the Vulkan compute device");
        }
        vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
        immediate_.create(physicalDevice_, device_, queueFamily_, queue_);
    } catch (...) {
        cleanup();
        throw;
    }
}

HeadlessComputeContext::~HeadlessComputeContext() { cleanup(); }

void HeadlessComputeContext::waitIdle() const {
    if (device_ == VK_NULL_HANDLE || vkDeviceWaitIdle(device_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to wait for the headless compute device");
    }
}

void HeadlessComputeContext::cleanup() {
    if (device_ != VK_NULL_HANDLE) {
        static_cast<void>(vkDeviceWaitIdle(device_));
    }
    immediate_.reset();
    queue_ = VK_NULL_HANDLE;
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    physicalDevice_ = VK_NULL_HANDLE;
    deviceName_.clear();
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

} // namespace vkexp
