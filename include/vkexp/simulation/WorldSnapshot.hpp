#pragma once

#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/simulation/AgentTypes.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

namespace vkexp {

class WorldSnapshotError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A whole experiment frozen mid-flight: the population, where every agent is,
// how far into the generation it got, and the settings that produced all of it.
// A genome archive holds only the weights, which is the right thing to carry
// between runs; this is the right thing to carry between sessions.
//
// The trail field is deliberately not here. It is device-local, it is the
// largest thing in the simulation by two orders of magnitude, and it is derived:
// a couple of half-lives of stepping rebuilds it. Storing it would make a
// snapshot a hundred times bigger to save a few seconds of simulation.
struct WorldSnapshot {
    SimulationStep physics{};
    std::vector<Genome> genomes;
    std::vector<AgentState> agents;
    std::uint64_t generation{};
    std::uint32_t step{};
    std::uint32_t stepsPerGeneration{};
    std::uint32_t requestedAgentsPerWorld{};
    std::uint32_t trialsPerGenome{};
    std::uint32_t seed{};
};

// 2 added the group fitness sharing weight. A version 1 file describes a run
// that had no such setting, which is not the same as one that had it at zero --
// so it is rejected rather than defaulted.
inline constexpr std::uint32_t worldSnapshotVersion = 2;

// Versioned and little-endian, like the genome archive, and just as strict: a
// file from another brain topology, another agent layout or another scenario
// count is rejected rather than reinterpreted.
void saveWorldSnapshot(const std::filesystem::path& path, const WorldSnapshot& snapshot);

[[nodiscard]] WorldSnapshot loadWorldSnapshot(const std::filesystem::path& path);

} // namespace vkexp
