#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace vkexp {

struct EvolutionSettings {
    std::size_t populationSize{512};
    std::size_t eliteCount{12};
    std::size_t tournamentSize{5};
    float crossoverProbability{0.65F};
    float mutationProbability{0.08F};
    float mutationStrength{0.18F};
    std::uint32_t seed{0xC0FFEEU};
};

struct Genome {
    neuro::Weights weights{};
};

struct GenerationSummary {
    std::uint64_t generation{};
    float bestFitness{};
    float meanFitness{};
    float medianFitness{};
    std::size_t championIndex{};
};

class GeneticAlgorithm {
public:
    explicit GeneticAlgorithm(EvolutionSettings settings = {});

    void reset();
    [[nodiscard]] GenerationSummary evolve(std::span<const float> fitness);

    // Replaces the population, e.g. when resuming from a genome archive. The
    // count must match populationSize so buffer sizes stay valid.
    void setPopulation(std::span<const Genome> genomes, std::uint64_t generation);

    [[nodiscard]] const std::vector<Genome>& population() const { return population_; }
    [[nodiscard]] std::uint64_t generation() const { return generation_; }
    [[nodiscard]] const EvolutionSettings& settings() const { return settings_; }

private:
    [[nodiscard]] std::size_t tournament(std::span<const float> fitness);

    EvolutionSettings settings_;
    std::mt19937 random_;
    std::vector<Genome> population_;
    std::uint64_t generation_{};
};

} // namespace vkexp
