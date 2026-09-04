#pragma once

#include "vkexp/profiling/ProfilerTypes.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>

namespace vkexp {

class VulkanContext;

class GpuProfiler {
public:
    class Scope {
    public:
        Scope(GpuProfiler& profiler, VkCommandBuffer commands, ProfileMetricId metric);
        ~Scope();

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        GpuProfiler* profiler_{};
        VkCommandBuffer commands_{};
        ProfileMetricId metric_{invalidProfileMetric};
    };

    explicit GpuProfiler(VulkanContext& vulkan);
    ~GpuProfiler();

    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;

    void beginFrame(VkCommandBuffer commands, ProfileMetricId frameMetric);
    void endFrame(VkCommandBuffer commands, ProfileMetricId frameMetric);
    [[nodiscard]] Scope scope(VkCommandBuffer commands, ProfileMetricId metric) {
        return Scope{*this, commands, metric};
    }

    [[nodiscard]] bool supported() const { return supported_; }
    [[nodiscard]] const TimingSeries& series(ProfileMetricId metric) const {
        return series_[metric];
    }

private:
    friend class Scope;

    static constexpr std::uint32_t queryCount = static_cast<std::uint32_t>(maxProfileMetrics * 2U);

    void writeBegin(VkCommandBuffer commands, ProfileMetricId metric);
    void writeEnd(VkCommandBuffer commands, ProfileMetricId metric);
    void resolvePendingFrame();

    VkDevice device_{};
    VkQueryPool queryPool_{};
    float timestampPeriodNs_{};
    std::uint32_t timestampValidBits_{};
    std::array<bool, maxProfileMetrics> currentWritten_{};
    std::array<bool, maxProfileMetrics> pendingWritten_{};
    std::array<TimingSeries, maxProfileMetrics> series_{};
    bool supported_{};
    bool pendingFrame_{};
};

} // namespace vkexp
