#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"

#include <span>

namespace vkexp {

void stepAgentCpu(AgentState& agent, std::span<const float, neuro::Topology::weightCount> weights,
                  const SimulationStep& settings);

[[nodiscard]] float agentFitness(const AgentState& agent,
                                 BeaconScenario scenario = BeaconScenario::Stationary);

} // namespace vkexp
