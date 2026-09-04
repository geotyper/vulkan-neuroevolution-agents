#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"

namespace vkexp {

// Pure CPU sensor model matching agent_step.comp. Keeping perception outside
// the neural network makes later wall, sound, pheromone, or spatial-grid
// implementations replaceable without changing the brain representation.
[[nodiscard]] neuro::Inputs samplePhotoreceptors(const AgentState& agent,
                                                 const SimulationStep& settings);

} // namespace vkexp
