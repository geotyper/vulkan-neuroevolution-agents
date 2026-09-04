#pragma once

#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/simulation/AgentTypes.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vkexp {

struct SimulationControls {
    bool paused{};
    bool resetRequested{};
    std::uint32_t stepsPerFrame{4};
    std::uint32_t stepsPerGeneration{900};
};

struct SimulationStatistics {
    std::uint64_t generation{};
    std::uint32_t step{};
    float bestFitness{};
    float meanFitness{};
    float medianFitness{};
    float arrivalRatio{};
};

struct EvolutionHistory {
    std::vector<float> bestFitness;
    std::vector<float> medianFitness;
    std::vector<float> meanFitness;
    std::vector<float> arrivalRatio;
    std::size_t maximumSamples{256};
};

struct AgentBufferView {
    VkBuffer buffer{};
    VkDeviceSize size{};
    std::uint32_t agentCount{};
    std::uint32_t genomeCount{};
    std::uint32_t trialsPerGenome{};
    std::uint64_t generation{};
};

struct SimulationViewport {
    VkImageView imageView{};
    VkSampler sampler{};
    VkExtent2D extent{960, 720};
    std::uint32_t requestedWidth{960};
    std::uint32_t requestedHeight{720};
    std::uint64_t generation{};
};

struct SimulationState {
    SimulationControls controls;
    SimulationStatistics statistics;
    EvolutionSettings evolution;
    EvolutionHistory history;
    SimulationStep physics;
    AgentBufferView agents;
    SimulationViewport viewport;
};

} // namespace vkexp
