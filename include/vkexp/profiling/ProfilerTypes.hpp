#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

namespace vkexp {

using ProfileMetricId = std::uint32_t;
inline constexpr ProfileMetricId invalidProfileMetric = ~ProfileMetricId{0};
inline constexpr std::size_t maxProfileMetrics = 32;

struct TimingStatistics {
    float currentMs{};
    float averageMs{};
    float minimumMs{};
    float maximumMs{};
    float percentile95Ms{};
    std::size_t sampleCount{};
};

class TimingSeries {
public:
    static constexpr std::size_t capacity = 120;

    void add(const double milliseconds) {
        const float value = static_cast<float>(milliseconds);
        if (count_ < capacity) {
            values_[count_++] = value;
            return;
        }
        std::move(values_.begin() + 1, values_.end(), values_.begin());
        values_.back() = value;
    }

    [[nodiscard]] std::span<const float> values() const { return {values_.data(), count_}; }

    [[nodiscard]] TimingStatistics statistics() const {
        TimingStatistics result{};
        result.sampleCount = count_;
        if (count_ == 0) {
            return result;
        }

        const auto samples = values();
        result.currentMs = samples.back();
        const auto [minimum, maximum] = std::minmax_element(samples.begin(), samples.end());
        result.minimumMs = *minimum;
        result.maximumMs = *maximum;
        result.averageMs = std::accumulate(samples.begin(), samples.end(), 0.0F) /
                           static_cast<float>(samples.size());

        std::vector<float> sorted{samples.begin(), samples.end()};
        std::sort(sorted.begin(), sorted.end());
        const std::size_t percentileIndex = ((sorted.size() * 95U + 99U) / 100U) - 1U;
        result.percentile95Ms = sorted[percentileIndex];
        return result;
    }

private:
    std::array<float, capacity> values_{};
    std::size_t count_{};
};

} // namespace vkexp
