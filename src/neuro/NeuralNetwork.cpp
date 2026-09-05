#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cmath>
#include <stdexcept>

namespace vkexp::neuro {

Outputs evaluate(const std::span<const float, Topology::weightCount> weights, const Inputs& inputs,
                 const BrainShape shape) {
    if (!shape.fitsCapacity()) {
        throw std::invalid_argument("Neural-network shape exceeds genome capacity");
    }
    std::array<float, Topology::hiddenCount> hidden{};
    std::size_t offset = shape.inputCount * shape.hiddenCount;
    for (std::size_t neuron = 0; neuron < shape.hiddenCount; ++neuron) {
        float activation = weights[offset + neuron];
        for (std::size_t input = 0; input < shape.inputCount; ++input) {
            activation += weights[neuron * shape.inputCount + input] * inputs[input];
        }
        hidden[neuron] = std::tanh(activation);
    }

    offset += shape.hiddenCount;
    const std::size_t outputBias = offset + shape.hiddenCount * shape.outputCount;
    Outputs outputs{};
    for (std::size_t neuron = 0; neuron < shape.outputCount; ++neuron) {
        float activation = weights[outputBias + neuron];
        for (std::size_t input = 0; input < shape.hiddenCount; ++input) {
            activation += weights[offset + neuron * shape.hiddenCount + input] * hidden[input];
        }
        outputs[neuron] = std::tanh(activation);
    }
    return outputs;
}

} // namespace vkexp::neuro
