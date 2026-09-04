#pragma once

#include "vkexp/presets/Preset.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <utility>

namespace vkexp {

struct RenderViewport {
    VkImage image{};
    VkImageView imageView{};
    VkSampler sampler{};
    VkExtent2D extent{960, 540};
    std::uint32_t requestedWidth{960};
    std::uint32_t requestedHeight{540};
    std::uint64_t generation{};
};

struct ComputeOutput {
    VkImageView imageView{};
    VkSampler sampler{};
    VkExtent2D extent{};
    std::uint64_t generation{};
    bool requested{};
    bool ready{};
    int radius{4};
};

struct DemoState {
    explicit DemoState(Preset selectedPreset) : preset(std::move(selectedPreset)) {}

    Preset preset;
    RenderViewport viewport;
    ComputeOutput blur;
};

} // namespace vkexp
