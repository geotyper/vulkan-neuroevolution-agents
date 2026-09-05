#pragma once

#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkexp {

class GenomeArchiveError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Provenance stored alongside the weights so a resumed run can be traced back
// to the experiment that produced it.
struct GenomeArchiveMetadata {
    std::uint64_t generation{};
    std::uint32_t scenario{};
    std::uint32_t seed{};
    float bestFitness{};
    float meanFitness{};
    std::uint32_t brainInputCount{};
    std::uint32_t brainHiddenCount{};
    std::uint32_t brainOutputCount{};
};

struct GenomeArchive {
    GenomeArchiveMetadata metadata;
    std::vector<Genome> genomes;
};

inline constexpr std::uint32_t genomeArchiveVersion = 1;

// Writes a versioned little-endian archive. Weight count and brain shape are
// recorded so a file from an incompatible topology fails loudly on load rather
// than being silently reinterpreted.
void saveGenomeArchive(const std::filesystem::path& path, std::span<const Genome> genomes,
                       const GenomeArchiveMetadata& metadata);

[[nodiscard]] GenomeArchive loadGenomeArchive(const std::filesystem::path& path);

} // namespace vkexp
