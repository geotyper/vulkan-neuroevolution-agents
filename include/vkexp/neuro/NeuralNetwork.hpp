#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace vkexp::neuro {

struct Topology {
    static constexpr std::size_t lightReceptorCount = 7;
    static constexpr std::size_t lightChannelsPerReceptor = 4; // RGB + luminance
    static constexpr std::size_t tactileSectorCount = 8;
    static constexpr std::size_t tactileChannelsPerSector = 2; // wall + agent
    static constexpr std::size_t selfInputCount = 4;
    static constexpr std::size_t inputCount = lightReceptorCount * lightChannelsPerReceptor +
                                              tactileSectorCount * tactileChannelsPerSector +
                                              selfInputCount;
    static constexpr std::size_t hiddenCount = 20;
    static constexpr std::size_t outputCount = 6;
    static constexpr std::size_t weightCount =
        inputCount * hiddenCount + hiddenCount + hiddenCount * outputCount + outputCount;
};

using Inputs = std::array<float, Topology::inputCount>;
using Outputs = std::array<float, Topology::outputCount>;
using Weights = std::array<float, Topology::weightCount>;

// Reference implementation for tests, tools, and deterministic inspection.
// The GPU shader deliberately uses the same flattened weight layout.
[[nodiscard]] Outputs evaluate(std::span<const float, Topology::weightCount> weights,
                               const Inputs& inputs);

} // namespace vkexp::neuro
