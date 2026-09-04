#pragma once

#include "vkexp/profiling/ProfilerTypes.hpp"

#include <array>
#include <chrono>
#include <ctime>

namespace vkexp {

class CpuProfiler {
public:
    struct FrameSample {
        double wallMilliseconds{};
        double cpuMilliseconds{};
    };

    class Scope {
    public:
        Scope(CpuProfiler& profiler, ProfileMetricId metric);
        ~Scope();

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        CpuProfiler* profiler_{};
        ProfileMetricId metric_{invalidProfileMetric};
        std::chrono::steady_clock::time_point started_{};
    };

    void beginFrame();
    [[nodiscard]] FrameSample endFrame(ProfileMetricId frameMetric, ProfileMetricId cpuWorkMetric,
                                       std::size_t metricCount);
    void addDuration(ProfileMetricId metric, double milliseconds);
    [[nodiscard]] Scope scope(ProfileMetricId metric) { return Scope{*this, metric}; }
    [[nodiscard]] const TimingSeries& series(ProfileMetricId metric) const {
        return series_[metric];
    }

private:
    friend class Scope;

    using Clock = std::chrono::steady_clock;
    Clock::time_point frameStarted_{};
    std::array<double, maxProfileMetrics> accumulatedMs_{};
    std::clock_t cpuStarted_{};
    std::array<TimingSeries, maxProfileMetrics> series_{};
    bool frameActive_{};
};

} // namespace vkexp
