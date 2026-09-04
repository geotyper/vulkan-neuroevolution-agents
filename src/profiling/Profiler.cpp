#include "vkexp/profiling/Profiler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vkexp {

Profiler::Profiler(VulkanContext& vulkan) : gpu_(vulkan) {
    cpuFrameMetric_ = registerMetric("Frame wall");
    cpuWorkMetric_ = registerMetric("CPU work");
    fenceWaitMetric_ = registerMetric("Fence wait");
    acquireWaitMetric_ = registerMetric("Acquire wait");
    queueSubmitMetric_ = registerMetric("Queue submit");
    presentMetric_ = registerMetric("Present");
    gpuFrameMetric_ = registerMetric("GPU frame");
}

ProfileMetricId Profiler::registerMetric(const std::string_view name) {
    if (name.empty()) {
        throw std::invalid_argument("Profiler metric name cannot be empty");
    }
    const auto found = std::find(metricNames_.begin(), metricNames_.end(), name);
    if (found != metricNames_.end()) {
        return static_cast<ProfileMetricId>(std::distance(metricNames_.begin(), found));
    }
    if (metricNames_.size() >= maxProfileMetrics) {
        throw std::length_error("Profiler metric capacity exceeded");
    }
    const auto metric = static_cast<ProfileMetricId>(metricNames_.size());
    metricNames_.emplace_back(name);
    return metric;
}

void Profiler::endCpuFrame(const double refreshRateHz) {
    const CpuProfiler::FrameSample sample =
        cpu_.endFrame(cpuFrameMetric_, cpuWorkMetric_, metricNames_.size());
    frameDiagnostics_.refreshRateHz = refreshRateHz;
    frameDiagnostics_.currentCpuLoadPercent =
        sample.wallMilliseconds > 0.0 ? sample.cpuMilliseconds / sample.wallMilliseconds * 100.0
                                      : 0.0;
    frameDiagnostics_.estimatedMissedVsyncs = 0;
    if (refreshRateHz > 0.0 && sample.wallMilliseconds > 0.0) {
        const double refreshIntervalMs = 1000.0 / refreshRateHz;
        const auto intervals = static_cast<std::uint64_t>(
            std::max(1.0, std::round(sample.wallMilliseconds / refreshIntervalMs)));
        frameDiagnostics_.estimatedMissedVsyncs =
            static_cast<std::uint32_t>(intervals > 1 ? intervals - 1 : 0);
        frameDiagnostics_.totalEstimatedMissedVsyncs += frameDiagnostics_.estimatedMissedVsyncs;
    }
}

void Profiler::recordFrameSynchronization(const double fenceWaitMs, const double acquireWaitMs,
                                          const double queueSubmitMs, const double presentMs) {
    cpu_.addDuration(fenceWaitMetric_, fenceWaitMs);
    cpu_.addDuration(acquireWaitMetric_, acquireWaitMs);
    cpu_.addDuration(queueSubmitMetric_, queueSubmitMs);
    cpu_.addDuration(presentMetric_, presentMs);
}

} // namespace vkexp
