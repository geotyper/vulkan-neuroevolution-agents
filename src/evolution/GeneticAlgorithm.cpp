#include "vkexp/evolution/GeneticAlgorithm.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace vkexp {

GeneticAlgorithm::GeneticAlgorithm(EvolutionSettings settings)
    : settings_(settings), random_(settings.seed) {
    if (settings_.populationSize < 2 || settings_.eliteCount == 0 ||
        settings_.eliteCount >= settings_.populationSize || settings_.tournamentSize == 0) {
        throw std::invalid_argument("Invalid genetic algorithm population settings");
    }
    reset();
}

void GeneticAlgorithm::reset() {
    random_.seed(settings_.seed);
    generation_ = 0;
    population_.assign(settings_.populationSize, {});
    std::normal_distribution<float> initialWeight{0.0F, 0.55F};
    for (Genome& genome : population_) {
        for (float& weight : genome.weights) {
            weight = initialWeight(random_);
        }
    }
}

void GeneticAlgorithm::setPopulation(const std::span<const Genome> genomes,
                                     const std::uint64_t generation) {
    if (genomes.size() != population_.size()) {
        throw std::invalid_argument("Loaded genome count must match the configured population");
    }
    population_.assign(genomes.begin(), genomes.end());
    generation_ = generation;
}

std::size_t GeneticAlgorithm::tournament(const std::span<const float> fitness) {
    std::uniform_int_distribution<std::size_t> candidate{0, fitness.size() - 1};
    std::size_t best = candidate(random_);
    for (std::size_t round = 1; round < settings_.tournamentSize; ++round) {
        const std::size_t next = candidate(random_);
        if (fitness[next] > fitness[best]) {
            best = next;
        }
    }
    return best;
}

GenerationSummary GeneticAlgorithm::evolve(const std::span<const float> fitness) {
    if (fitness.size() != population_.size()) {
        throw std::invalid_argument("Fitness count must match the genome population");
    }
    std::vector<std::size_t> ranking(fitness.size());
    std::iota(ranking.begin(), ranking.end(), 0);
    std::stable_sort(ranking.begin(), ranking.end(), [&](const std::size_t a, const std::size_t b) {
        return fitness[a] > fitness[b];
    });
    std::vector<float> sortedFitness(fitness.begin(), fitness.end());
    std::sort(sortedFitness.begin(), sortedFitness.end());
    const float sum = std::accumulate(fitness.begin(), fitness.end(), 0.0F);
    GenerationSummary summary{generation_, fitness[ranking.front()],
                              sum / static_cast<float>(fitness.size()),
                              sortedFitness[sortedFitness.size() / 2], ranking.front()};

    std::vector<Genome> next(population_.size());
    for (std::size_t index = 0; index < settings_.eliteCount; ++index) {
        next[index] = population_[ranking[index]];
    }

    std::bernoulli_distribution crossover(settings_.crossoverProbability);
    std::bernoulli_distribution inheritSecond(0.5);
    std::bernoulli_distribution mutate(settings_.mutationProbability);
    std::normal_distribution<float> mutation{0.0F, settings_.mutationStrength};
    for (std::size_t index = settings_.eliteCount; index < next.size(); ++index) {
        const Genome& first = population_[tournament(fitness)];
        const Genome& second = population_[tournament(fitness)];
        const bool useCrossover = crossover(random_);
        for (std::size_t weight = 0; weight < neuro::Topology::weightCount; ++weight) {
            float value = first.weights[weight];
            if (useCrossover && inheritSecond(random_)) {
                value = second.weights[weight];
            }
            if (mutate(random_)) {
                value += mutation(random_);
            }
            next[index].weights[weight] = std::clamp(value, -6.0F, 6.0F);
        }
    }
    population_ = std::move(next);
    ++generation_;
    return summary;
}

} // namespace vkexp
