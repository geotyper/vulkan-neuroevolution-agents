#include "vkexp/profiling/CpuProfiler.hpp"

#include <utility>

namespace vkexp {

CpuProfiler::Scope::Scope(CpuProfiler& profiler, const ProfileMetricId metric)
    : profiler_(&profiler), metric_(metric), started_(std::chrono::steady_clock::now()) {}

CpuProfiler::Scope::~Scope() {
    if (profiler_ == nullptr) {
        return;
    }
    const auto elapsed = std::chrono::steady_clock::now() - started_;
    profiler_->addDuration(metric_, std::chrono::duration<double, std::milli>(elapsed).count());
}

CpuProfiler::Scope::Scope(Scope&& other) noexcept
    : profiler_(std::exchange(other.profiler_, nullptr)), metric_(other.metric_),
      started_(other.started_) {}

void CpuProfiler::beginFrame() {
    accumulatedMs_.fill(0.0);
    frameStarted_ = Clock::now();
    cpuStarted_ = std::clock();
    frameActive_ = true;
}

CpuProfiler::FrameSample CpuProfiler::endFrame(const ProfileMetricId frameMetric,
                                               const ProfileMetricId cpuWorkMetric,
                                               const std::size_t metricCount) {
    if (!frameActive_) {
        return {};
    }
    FrameSample sample;
    sample.wallMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - frameStarted_).count();
    const std::clock_t cpuFinished = std::clock();
    if (cpuStarted_ != static_cast<std::clock_t>(-1) &&
        cpuFinished != static_cast<std::clock_t>(-1)) {
        sample.cpuMilliseconds = static_cast<double>(cpuFinished - cpuStarted_) * 1000.0 /
                                 static_cast<double>(CLOCKS_PER_SEC);
    }
    accumulatedMs_[frameMetric] = sample.wallMilliseconds;
    accumulatedMs_[cpuWorkMetric] = sample.cpuMilliseconds;
    for (std::size_t index = 0; index < metricCount; ++index) {
        if (index != frameMetric && index != cpuWorkMetric && accumulatedMs_[index] == 0.0) {
            continue;
        }
        series_[index].add(accumulatedMs_[index]);
    }
    frameActive_ = false;
    return sample;
}

void CpuProfiler::addDuration(const ProfileMetricId metric, const double milliseconds) {
    if (frameActive_ && metric < maxProfileMetrics) {
        accumulatedMs_[metric] += milliseconds;
    }
}

} // namespace vkexp
