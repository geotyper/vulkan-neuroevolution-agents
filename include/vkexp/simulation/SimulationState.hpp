#pragma once

#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/simulation/AgentTypes.hpp"

#include <vulkan/vulkan.h>

#include <array>
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

// What the viewport draws. None of this reaches the simulation -- turning the
// agents off to watch the bare trail field changes the picture, not the run.
struct SimulationDisplay {
    bool agents{true};
    bool beacons{true};
    bool trail{true};
    // Discs read as a track, squares tile the cell exactly and show the grid the
    // field really is. Both are worth having in front of you, so this stays a
    // switch rather than a decision.
    bool roundTrailMarks{true};
};

struct SimulationWorlds {
    std::uint32_t requestedAgentsPerWorld{12};
    std::uint32_t agentsPerWorld{12};
    std::uint32_t groupCount{};
    std::uint32_t worldCount{};
    std::uint32_t selectedWorld{};
};

struct AgentBufferView {
    std::array<VkBuffer, 2> buffers{};
    VkDeviceSize size{};
    std::uint32_t currentIndex{};
    std::uint32_t agentCount{};
    std::uint32_t genomeCount{};
    std::uint32_t trialsPerGenome{};
    std::uint64_t generation{};

    [[nodiscard]] VkBuffer currentBuffer() const { return buffers[currentIndex]; }
};

// Published so the renderer can draw the field it never writes. Sized once for
// the largest arena and the most logical worlds the population can be split
// into, so the buffer never reallocates and this handle never goes stale.
struct TrailBufferView {
    VkBuffer buffer{};
    VkDeviceSize size{};
    std::uint32_t width{};
    std::uint32_t cellsPerWorld{};
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
    SimulationWorlds worlds;
    SimulationStep physics;
    SimulationDisplay display;
    AgentBufferView agents;
    TrailBufferView trail;
    SimulationViewport viewport;
};

} // namespace vkexp
