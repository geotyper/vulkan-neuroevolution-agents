#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace vkexp::neuro {

struct Topology {
    static constexpr std::size_t receptorCount = 7;
    static constexpr std::size_t inputCount = 10;
    static constexpr std::size_t hiddenCount = 12;
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
