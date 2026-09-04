#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/presets/PresetRegistry.hpp"
#include "vkexp/profiling/CpuProfiler.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <string_view>

namespace {

int failures = 0;

void check(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

bool closeTo(const float left, const float right) { return std::abs(left - right) < 0.0001F; }

void testTimingSeries() {
    vkexp::TimingSeries series;
    series.add(1.0);
    series.add(2.0);
    series.add(3.0);
    series.add(4.0);

    const auto statistics = series.statistics();
    check(statistics.sampleCount == 4, "TimingSeries sample count");
    check(closeTo(statistics.currentMs, 4.0F), "TimingSeries current value");
    check(closeTo(statistics.averageMs, 2.5F), "TimingSeries average");
    check(closeTo(statistics.minimumMs, 1.0F), "TimingSeries minimum");
    check(closeTo(statistics.maximumMs, 4.0F), "TimingSeries maximum");
    check(closeTo(statistics.percentile95Ms, 4.0F), "TimingSeries p95");
}

void testCpuProfiler() {
    constexpr vkexp::ProfileMetricId frameMetric = 0;
    constexpr vkexp::ProfileMetricId cpuWorkMetric = 1;
    constexpr vkexp::ProfileMetricId customMetric = 2;
    vkexp::CpuProfiler profiler;
    profiler.beginFrame();
    profiler.addDuration(customMetric, 1.25);
    const vkexp::CpuProfiler::FrameSample sample = profiler.endFrame(frameMetric, cpuWorkMetric, 3);

    check(sample.wallMilliseconds >= 0.0, "CPU profiler wall time");
    check(sample.cpuMilliseconds >= 0.0, "CPU profiler process time");
    check(profiler.series(frameMetric).statistics().sampleCount == 1, "CPU frame sample");
    check(profiler.series(cpuWorkMetric).statistics().sampleCount == 1, "CPU work sample");
    check(closeTo(profiler.series(customMetric).statistics().currentMs, 1.25F),
          "CPU custom duration");
}

void testPresetRegistry() {
    vkexp::PresetRegistry registry;
    check(registry.all().size() == 3, "Built-in preset count");
    check(registry.require("mixed").graphicsEnabled, "Mixed preset graphics");
    check(registry.require("mixed").computeEnabled, "Mixed preset compute");

    registry.loadWindowPreset(VKEXP_TEST_WINDOW_PRESET);
    for (const auto& preset : registry.all()) {
        check(preset.windowWidth >= 320 && preset.windowWidth <= 7680, "Loaded preset width range");
        check(preset.windowHeight >= 240 && preset.windowHeight <= 4320,
              "Loaded preset height range");
    }

    bool rejectedUnknown = false;
    try {
        static_cast<void>(registry.require("does-not-exist"));
    } catch (const std::exception&) {
        rejectedUnknown = true;
    }
    check(rejectedUnknown, "Unknown preset rejection");
}

void testDispatchSize() {
    check(vkexp::divideRoundUp(17, 8) == 3, "Rounded-up integer division");
    check(vkexp::divideRoundUp(16, 8) == 2, "Exact integer division");

    const vkexp::DispatchSize groups = vkexp::dispatchSize({1921, 1081, 1}, {8, 8, 1});
    check(groups.x == 241, "Dispatch width");
    check(groups.y == 136, "Dispatch height");
    check(groups.z == 1, "Dispatch depth");

    bool rejectedZero = false;
    try {
        static_cast<void>(vkexp::dispatchSize({1, 1, 1}, {0, 1, 1}));
    } catch (const std::exception&) {
        rejectedZero = true;
    }
    check(rejectedZero, "Zero local size rejection");

    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupCount[0] = 1024;
    limits.maxComputeWorkGroupCount[1] = 1024;
    limits.maxComputeWorkGroupCount[2] = 64;
    limits.maxComputeWorkGroupSize[0] = 1024;
    limits.maxComputeWorkGroupSize[1] = 1024;
    limits.maxComputeWorkGroupSize[2] = 64;
    limits.maxComputeWorkGroupInvocations = 1024;
    limits.maxPushConstantsSize = 128;
    limits.maxStorageBufferRange = 4096;
    const std::array<VkDeviceSize, 2> validRanges{1024, 2048};
    vkexp::validateComputeLimits(limits, groups, {8, 8, 1}, 16, validRanges);

    bool rejectedGroupCount = false;
    try {
        vkexp::validateComputeLimits(limits, {1025, 1, 1}, {8, 8, 1});
    } catch (const std::exception&) {
        rejectedGroupCount = true;
    }
    check(rejectedGroupCount, "Dispatch group limit rejection");

    bool rejectedInvocations = false;
    try {
        vkexp::validateComputeLimits(limits, {1, 1, 1}, {64, 64, 1});
    } catch (const std::exception&) {
        rejectedInvocations = true;
    }
    check(rejectedInvocations, "Local invocation limit rejection");

    bool rejectedPushConstants = false;
    try {
        vkexp::validateComputeLimits(limits, {1, 1, 1}, {8, 8, 1}, 132);
    } catch (const std::exception&) {
        rejectedPushConstants = true;
    }
    check(rejectedPushConstants, "Push constant limit rejection");

    const std::array<VkDeviceSize, 1> oversizedRange{8192};
    bool rejectedStorageRange = false;
    try {
        vkexp::validateComputeLimits(limits, {1, 1, 1}, {8, 8, 1}, 0, oversizedRange);
    } catch (const std::exception&) {
        rejectedStorageRange = true;
    }
    check(rejectedStorageRange, "Storage buffer range limit rejection");
}

void testComputeResourceValidation() {
    check(vkexp::tightlyPackedImageSize(VK_FORMAT_R8G8B8A8_UNORM, {4, 4}) == 64,
          "RGBA8 tightly-packed image size");
    check(vkexp::tightlyPackedImageSize(VK_FORMAT_R32G32_SFLOAT, {3, 2}) == 48,
          "RG32 tightly-packed image size");

    bool rejectedUnsupportedFormat = false;
    try {
        static_cast<void>(vkexp::tightlyPackedImageSize(VK_FORMAT_D32_SFLOAT, {4, 4}));
    } catch (const std::exception&) {
        rejectedUnsupportedFormat = true;
    }
    check(rejectedUnsupportedFormat, "Unsupported image transfer format rejection");

    vkexp::ComputePipelineBuilder builder{VK_NULL_HANDLE, VK_NULL_HANDLE};
    builder.specializationConstant(7, std::uint32_t{42});
    bool rejectedDuplicateConstant = false;
    try {
        builder.specializationConstant(7, std::uint32_t{43});
    } catch (const std::exception&) {
        rejectedDuplicateConstant = true;
    }
    check(rejectedDuplicateConstant, "Duplicate specialization constant rejection");

    bool rejectedUnavailableDescriptorSet = false;
    try {
        static_cast<void>(vkexp::PingPongDescriptorSets{}.forReadIndex(0));
    } catch (const std::exception&) {
        rejectedUnavailableDescriptorSet = true;
    }
    check(rejectedUnavailableDescriptorSet, "Unavailable ping-pong descriptor rejection");
}

void testPingPongState() {
    vkexp::PingPongBuffer buffers;
    check(buffers.readIndex() == 0 && buffers.writeIndex() == 1, "Initial ping-pong indices");
    buffers.swap();
    check(buffers.readIndex() == 1 && buffers.writeIndex() == 0, "Swapped ping-pong indices");
    buffers.swap();
    check(buffers.readIndex() == 0 && buffers.writeIndex() == 1, "Restored ping-pong indices");
}

} // namespace

int main() {
    testTimingSeries();
    testCpuProfiler();
    testPresetRegistry();
    testDispatchSize();
    testComputeResourceValidation();
    testPingPongState();
    return failures == 0 ? 0 : 1;
}
