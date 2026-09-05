#pragma once

#include "vkexp/simulation/AgentTypes.hpp"

#include <array>
#include <cstdint>

namespace vkexp::worlds {

inline constexpr float tau = 6.28318530718F;
inline constexpr std::array<Float4, 4> trialColors{
    Float4{0.20F, 0.85F, 1.00F, 0.0F}, Float4{1.00F, 0.35F, 0.75F, 0.0F},
    Float4{0.55F, 1.00F, 0.35F, 0.0F}, Float4{1.00F, 0.72F, 0.20F, 0.0F}};

[[nodiscard]] inline std::uint32_t hash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

[[nodiscard]] inline float random01(const std::uint32_t value) {
    return static_cast<float>(hash(value) & 0x00ffffffU) / 16777215.0F;
}

[[nodiscard]] inline float objectiveFitness(const AgentState& agent,
                                            const std::uint32_t completedObjectives) {
    return agent.metrics.w + (agent.metrics.x - agent.metrics.y) +
           static_cast<float>(completedObjectives) * 2.0F - agent.metrics.z * 0.002F -
           agent.penalties.x;
}

} // namespace vkexp::worlds
