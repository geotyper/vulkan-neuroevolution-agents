#pragma once

#include <cmath>
#include <cstdint>

namespace vkexp::trail::kernel {

// The smallest GLSL-shaped surface that lets TrailKernel.inl compile as C++,
// following ScenarioKernel.hpp. Deliberately minimal for the same reason.
using uint = std::uint32_t;

struct vec2 {
    float x{};
    float y{};

    constexpr vec2() = default;
    constexpr vec2(const float xValue, const float yValue) : x(xValue), y(yValue) {}
};

[[nodiscard]] inline float exp(const float value) { return std::exp(value); }

#define VKEXP_TRAIL_FN inline
#include "vkexp/simulation/TrailKernel.inl"
#undef VKEXP_TRAIL_FN

} // namespace vkexp::trail::kernel
