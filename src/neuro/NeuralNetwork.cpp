#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cmath>
#include <stdexcept>

namespace vkexp::neuro {

Outputs evaluate(const std::span<const float, Topology::weightCount> weights, const Inputs& inputs,
                 const BrainShape shape) {
    if (!shape.fitsCapacity()) {
        throw std::invalid_argument("Neural-network shape exceeds genome capacity");
    }
    const auto inputCount = static_cast<kernel::uint>(shape.inputCount);
    const auto hiddenCount = static_cast<kernel::uint>(shape.hiddenCount);
    const auto outputCount = static_cast<kernel::uint>(shape.outputCount);
    constexpr kernel::uint base = 0; // one genome, so it starts at zero here

    std::array<float, Topology::hiddenCount> hidden{};
    for (kernel::uint neuron = 0; neuron < hiddenCount; ++neuron) {
        float activation =
            weights[kernel::brainHiddenBiasIndex(base, inputCount, hiddenCount, neuron)];
        for (kernel::uint input = 0; input < inputCount; ++input) {
            activation += weights[kernel::brainHiddenWeightIndex(base, inputCount, neuron, input)] *
                          inputs[input];
        }
        hidden[neuron] = std::tanh(activation);
    }

    Outputs outputs{};
    for (kernel::uint neuron = 0; neuron < outputCount; ++neuron) {
        float activation = weights[kernel::brainOutputBiasIndex(base, inputCount, hiddenCount,
                                                                outputCount, neuron)];
        for (kernel::uint hiddenIndex = 0; hiddenIndex < hiddenCount; ++hiddenIndex) {
            activation += weights[kernel::brainOutputWeightIndex(base, inputCount, hiddenCount,
                                                                 neuron, hiddenIndex)] *
                          hidden[hiddenIndex];
        }
        outputs[neuron] = std::tanh(activation);
    }
    return outputs;
}

} // namespace vkexp::neuro
