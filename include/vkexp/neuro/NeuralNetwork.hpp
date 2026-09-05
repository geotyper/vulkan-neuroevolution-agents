#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexp::neuro {

struct Topology {
    static constexpr std::size_t lightReceptorCount = 7;
    static constexpr std::size_t lightChannelsPerReceptor = 4; // RGB + luminance
    static constexpr std::size_t tactileSectorCount = 8;
    static constexpr std::size_t tactileChannelsPerSector = 2; // wall + agent
    static constexpr std::size_t selfInputCount = 4;
    static constexpr std::size_t taskInputCount = 2; // cargo level + seeking-home state
    static constexpr std::size_t recurrentMemoryCount = 2;
    static constexpr std::size_t inputCount = lightReceptorCount * lightChannelsPerReceptor +
                                              tactileSectorCount * tactileChannelsPerSector +
                                              selfInputCount + taskInputCount +
                                              recurrentMemoryCount;
    static constexpr std::size_t hiddenCount = 20;
    static constexpr std::size_t actuatorOutputCount = 6;
    static constexpr std::size_t outputCount = actuatorOutputCount + recurrentMemoryCount;
    static constexpr std::size_t weightCount =
        inputCount * hiddenCount + hiddenCount + hiddenCount * outputCount + outputCount;
};

// A scenario selects an active dense-network shape inside the fixed-capacity genome.
// Keeping capacity separate from shape lets GPU buffers and the GA stay reusable.
struct BrainShape {
    std::size_t inputCount{};
    std::size_t hiddenCount{};
    std::size_t outputCount{};

    [[nodiscard]] constexpr std::size_t weightCount() const {
        return inputCount * hiddenCount + hiddenCount + hiddenCount * outputCount + outputCount;
    }

    [[nodiscard]] constexpr bool fitsCapacity() const {
        return inputCount > 0 && inputCount <= Topology::inputCount && hiddenCount > 0 &&
               hiddenCount <= Topology::hiddenCount &&
               outputCount >= Topology::actuatorOutputCount &&
               outputCount <= Topology::outputCount && weightCount() <= Topology::weightCount;
    }
};

inline constexpr BrainShape maximumBrainShape{Topology::inputCount, Topology::hiddenCount,
                                              Topology::outputCount};

inline constexpr std::uint32_t brainStrideMask = 0x00000fffU;
inline constexpr std::uint32_t brainInputShift = 12U;
inline constexpr std::uint32_t brainHiddenShift = 18U;
inline constexpr std::uint32_t brainOutputShift = 24U;

[[nodiscard]] constexpr std::uint32_t
packBrainLayout(const BrainShape shape, const std::size_t genomeStride = Topology::weightCount) {
    return static_cast<std::uint32_t>(genomeStride) |
           (static_cast<std::uint32_t>(shape.inputCount) << brainInputShift) |
           (static_cast<std::uint32_t>(shape.hiddenCount) << brainHiddenShift) |
           (static_cast<std::uint32_t>(shape.outputCount) << brainOutputShift);
}

[[nodiscard]] constexpr std::size_t brainGenomeStride(const std::uint32_t layout) {
    return layout & brainStrideMask;
}

[[nodiscard]] constexpr BrainShape brainShape(const std::uint32_t layout) {
    return {(layout >> brainInputShift) & 0x3fU, (layout >> brainHiddenShift) & 0x3fU,
            (layout >> brainOutputShift) & 0x1fU};
}

static_assert(maximumBrainShape.fitsCapacity());
static_assert(Topology::weightCount <= brainStrideMask);

using Inputs = std::array<float, Topology::inputCount>;
using Outputs = std::array<float, Topology::outputCount>;
using Weights = std::array<float, Topology::weightCount>;

// Reference implementation for tests, tools, and deterministic inspection.
// The GPU shader deliberately uses the same flattened weight layout.
[[nodiscard]] Outputs evaluate(std::span<const float, Topology::weightCount> weights,
                               const Inputs& inputs, BrainShape shape = maximumBrainShape);

} // namespace vkexp::neuro
