#pragma once

#include "vkexp/profiling/CpuProfiler.hpp"
#include "vkexp/profiling/GpuProfiler.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vkexp {

class VulkanContext;
struct FrameDiagnostics {
    double refreshRateHz{};
    double currentCpuLoadPercent{};
    std::uint32_t estimatedMissedVsyncs{};
    std::uint64_t totalEstimatedMissedVsyncs{};
};

class Profiler {
public:
    explicit Profiler(VulkanContext& vulkan);

    [[nodiscard]] ProfileMetricId registerMetric(std::string_view name);
    [[nodiscard]] std::span<const std::string> metricNames() const { return metricNames_; }

    [[nodiscard]] CpuProfiler& cpu() { return cpu_; }
    [[nodiscard]] const CpuProfiler& cpu() const { return cpu_; }
    [[nodiscard]] GpuProfiler& gpu() { return gpu_; }
    [[nodiscard]] const GpuProfiler& gpu() const { return gpu_; }

    void beginCpuFrame() { cpu_.beginFrame(); }
    void endCpuFrame(double refreshRateHz);
    void beginGpuFrame(VkCommandBuffer commands) { gpu_.beginFrame(commands, gpuFrameMetric_); }
    void endGpuFrame(VkCommandBuffer commands) { gpu_.endFrame(commands, gpuFrameMetric_); }
    void recordFrameSynchronization(double fenceWaitMs, double acquireWaitMs, double queueSubmitMs,
                                    double presentMs);

    [[nodiscard]] const FrameDiagnostics& frameDiagnostics() const { return frameDiagnostics_; }
    [[nodiscard]] ProfileMetricId cpuFrameMetric() const { return cpuFrameMetric_; }
    [[nodiscard]] ProfileMetricId cpuWorkMetric() const { return cpuWorkMetric_; }

private:
    CpuProfiler cpu_;
    GpuProfiler gpu_;
    std::vector<std::string> metricNames_;
    ProfileMetricId cpuFrameMetric_{invalidProfileMetric};
    ProfileMetricId cpuWorkMetric_{invalidProfileMetric};
    ProfileMetricId fenceWaitMetric_{invalidProfileMetric};
    ProfileMetricId acquireWaitMetric_{invalidProfileMetric};
    ProfileMetricId queueSubmitMetric_{invalidProfileMetric};
    ProfileMetricId presentMetric_{invalidProfileMetric};
    ProfileMetricId gpuFrameMetric_{invalidProfileMetric};
    FrameDiagnostics frameDiagnostics_{};
};

} // namespace vkexp
