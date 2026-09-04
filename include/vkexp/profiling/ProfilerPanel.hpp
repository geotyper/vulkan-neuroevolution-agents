#pragma once

namespace vkexp {

class Profiler;

class ProfilerPanel {
public:
    void draw(const Profiler& profiler);

private:
    int selectedMetric_{};
};

} // namespace vkexp
