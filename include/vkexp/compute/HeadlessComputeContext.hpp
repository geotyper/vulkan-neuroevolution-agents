#pragma once

#include "vkexp/compute/ComputeResources.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace vkexp {

class HeadlessComputeUnavailable final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct HeadlessComputeConfig {
    std::string applicationName{"vkexp headless compute"};
    std::uint32_t apiVersion{VK_API_VERSION_1_3};
    bool preferDedicatedComputeQueue{true};
};

class HeadlessComputeContext {
public:
    explicit HeadlessComputeContext(const HeadlessComputeConfig& config = {});
    ~HeadlessComputeContext();

    HeadlessComputeContext(const HeadlessComputeContext&) = delete;
    HeadlessComputeContext& operator=(const HeadlessComputeContext&) = delete;
    HeadlessComputeContext(HeadlessComputeContext&&) = delete;
    HeadlessComputeContext& operator=(HeadlessComputeContext&&) = delete;

    void waitIdle() const;

    [[nodiscard]] VkInstance instance() const { return instance_; }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    [[nodiscard]] VkDevice device() const { return device_; }
    [[nodiscard]] VkQueue queue() const { return queue_; }
    [[nodiscard]] std::uint32_t queueFamily() const { return queueFamily_; }
    [[nodiscard]] const std::string& deviceName() const { return deviceName_; }
    [[nodiscard]] ImmediateContext& immediate() { return immediate_; }
    [[nodiscard]] const ImmediateContext& immediate() const { return immediate_; }

private:
    void cleanup();

    VkInstance instance_{};
    VkPhysicalDevice physicalDevice_{};
    VkDevice device_{};
    VkQueue queue_{};
    std::uint32_t queueFamily_{};
    std::string deviceName_;
    ImmediateContext immediate_;
};

} // namespace vkexp
