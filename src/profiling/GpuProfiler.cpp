#include "vkexp/profiling/GpuProfiler.hpp"

#include "vkexp/core/VulkanContext.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkexp {

GpuProfiler::Scope::Scope(GpuProfiler& profiler, const VkCommandBuffer commands,
                          const ProfileMetricId metric)
    : profiler_(&profiler), commands_(commands), metric_(metric) {
    profiler_->writeBegin(commands_, metric_);
}

GpuProfiler::Scope::~Scope() {
    if (profiler_ != nullptr) {
        profiler_->writeEnd(commands_, metric_);
    }
}

GpuProfiler::Scope::Scope(Scope&& other) noexcept
    : profiler_(std::exchange(other.profiler_, nullptr)), commands_(other.commands_),
      metric_(other.metric_) {}

GpuProfiler::GpuProfiler(VulkanContext& vulkan) : device_(vulkan.device()) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(vulkan.physicalDevice(), &properties);
    timestampPeriodNs_ = properties.limits.timestampPeriod;

    std::uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vulkan.physicalDevice(), &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(vulkan.physicalDevice(), &queueFamilyCount,
                                             queueFamilies.data());
    if (vulkan.graphicsQueueFamily() >= queueFamilies.size()) {
        return;
    }

    timestampValidBits_ = queueFamilies[vulkan.graphicsQueueFamily()].timestampValidBits;
    supported_ =
        timestampValidBits_ > 0 && properties.limits.timestampComputeAndGraphics == VK_TRUE;
    if (!supported_) {
        return;
    }

    VkQueryPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    poolInfo.queryCount = queryCount;
    if (vkCreateQueryPool(device_, &poolInfo, nullptr, &queryPool_) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create GPU timestamp query pool");
    }
}

GpuProfiler::~GpuProfiler() {
    if (queryPool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_, queryPool_, nullptr);
    }
}

void GpuProfiler::beginFrame(const VkCommandBuffer commands, const ProfileMetricId frameMetric) {
    if (!supported_) {
        return;
    }
    resolvePendingFrame();
    currentWritten_.fill(false);
    vkCmdResetQueryPool(commands, queryPool_, 0, queryCount);
    writeBegin(commands, frameMetric);
}

void GpuProfiler::endFrame(const VkCommandBuffer commands, const ProfileMetricId frameMetric) {
    if (!supported_) {
        return;
    }
    writeEnd(commands, frameMetric);
    pendingWritten_ = currentWritten_;
    pendingFrame_ = true;
}

void GpuProfiler::writeBegin(const VkCommandBuffer commands, const ProfileMetricId metric) {
    if (!supported_ || metric >= maxProfileMetrics) {
        return;
    }
    const std::uint32_t query = metric * 2U;
    vkCmdWriteTimestamp2(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queryPool_, query);
}

void GpuProfiler::writeEnd(const VkCommandBuffer commands, const ProfileMetricId metric) {
    if (!supported_ || metric >= maxProfileMetrics) {
        return;
    }
    const std::size_t index = metric;
    const std::uint32_t query = static_cast<std::uint32_t>(index * 2U + 1U);
    vkCmdWriteTimestamp2(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queryPool_, query);
    currentWritten_[index] = true;
}

void GpuProfiler::resolvePendingFrame() {
    if (!pendingFrame_) {
        return;
    }

    const std::uint64_t timestampMask = timestampValidBits_ >= 64U
                                            ? std::numeric_limits<std::uint64_t>::max()
                                            : (std::uint64_t{1} << timestampValidBits_) - 1U;

    for (std::size_t index = 0; index < maxProfileMetrics; ++index) {
        if (!pendingWritten_[index]) {
            continue;
        }
        std::array<std::uint64_t, 2> timestamps{};
        const std::uint32_t firstQuery = static_cast<std::uint32_t>(index * 2U);
        const VkResult result =
            vkGetQueryPoolResults(device_, queryPool_, firstQuery, 2, sizeof(timestamps),
                                  timestamps.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS) {
            continue;
        }
        const std::uint64_t ticks = (timestamps[1] - timestamps[0]) & timestampMask;
        const double milliseconds =
            static_cast<double>(ticks) * static_cast<double>(timestampPeriodNs_) / 1'000'000.0;
        series_[index].add(milliseconds);
    }
    pendingFrame_ = false;
}

} // namespace vkexp
