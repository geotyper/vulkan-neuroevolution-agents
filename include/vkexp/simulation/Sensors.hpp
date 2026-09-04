#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"

namespace vkexp {

// Pure CPU reference matching the isolated-agent path in agent_step.comp.
// Inter-agent radiance is supplied by the GPU spatial-grid pass at runtime.
[[nodiscard]] neuro::Inputs sampleAgentInputs(const AgentState& agent,
                                              const SimulationStep& settings);

} // namespace vkexp
