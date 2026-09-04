#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cmath>

namespace vkexp::neuro {

Outputs evaluate(const std::span<const float, Topology::weightCount> weights,
                 const Inputs& inputs) {
    std::array<float, Topology::hiddenCount> hidden{};
    std::size_t offset = Topology::inputCount * Topology::hiddenCount;
    for (std::size_t neuron = 0; neuron < Topology::hiddenCount; ++neuron) {
        float activation = weights[offset + neuron];
        for (std::size_t input = 0; input < Topology::inputCount; ++input) {
            activation += weights[neuron * Topology::inputCount + input] * inputs[input];
        }
        hidden[neuron] = std::tanh(activation);
    }

    offset += Topology::hiddenCount;
    const std::size_t outputBias = offset + Topology::hiddenCount * Topology::outputCount;
    Outputs outputs{};
    for (std::size_t neuron = 0; neuron < Topology::outputCount; ++neuron) {
        float activation = weights[outputBias + neuron];
        for (std::size_t input = 0; input < Topology::hiddenCount; ++input) {
            activation += weights[offset + neuron * Topology::hiddenCount + input] * hidden[input];
        }
        outputs[neuron] = std::tanh(activation);
    }
    return outputs;
}

} // namespace vkexp::neuro
