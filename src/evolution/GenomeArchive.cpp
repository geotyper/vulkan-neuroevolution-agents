#include "vkexp/evolution/GenomeArchive.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <fstream>

namespace vkexp {
namespace {

// Little-endian only for now; the version field lets a future reader adapt.
static_assert(std::endian::native == std::endian::little,
              "Genome archives are written little-endian");

constexpr std::array<char, 4> archiveMagic{'V', 'K', 'N', 'G'};

struct ArchiveHeader {
    std::array<char, 4> magic{};
    std::uint32_t version{};
    std::uint32_t genomeCount{};
    std::uint32_t weightCount{};
    std::uint32_t brainInputCount{};
    std::uint32_t brainHiddenCount{};
    std::uint32_t brainOutputCount{};
    std::uint32_t scenario{};
    std::uint64_t generation{};
    std::uint32_t seed{};
    float bestFitness{};
    float meanFitness{};
    std::uint32_t reserved{};
};

static_assert(sizeof(ArchiveHeader) == 56);

} // namespace

void saveGenomeArchive(const std::filesystem::path& path, const std::span<const Genome> genomes,
                       const GenomeArchiveMetadata& metadata) {
    if (genomes.empty()) {
        throw GenomeArchiveError("Refusing to write an empty genome archive");
    }
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code directoryError;
        std::filesystem::create_directories(parent, directoryError);
    }
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream) {
        throw GenomeArchiveError("Unable to open genome archive for writing: " + path.string());
    }
    const ArchiveHeader header{archiveMagic,
                               genomeArchiveVersion,
                               static_cast<std::uint32_t>(genomes.size()),
                               static_cast<std::uint32_t>(neuro::Topology::weightCount),
                               metadata.brainInputCount,
                               metadata.brainHiddenCount,
                               metadata.brainOutputCount,
                               metadata.scenario,
                               metadata.generation,
                               metadata.seed,
                               metadata.bestFitness,
                               metadata.meanFitness,
                               0U};
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (const Genome& genome : genomes) {
        stream.write(reinterpret_cast<const char*>(genome.weights.data()),
                     static_cast<std::streamsize>(genome.weights.size() * sizeof(float)));
    }
    stream.flush();
    if (!stream) {
        throw GenomeArchiveError("Failed while writing genome archive: " + path.string());
    }
}

GenomeArchive loadGenomeArchive(const std::filesystem::path& path) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        throw GenomeArchiveError("Unable to open genome archive: " + path.string());
    }
    ArchiveHeader header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream || stream.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        throw GenomeArchiveError("Genome archive is truncated: " + path.string());
    }
    if (header.magic != archiveMagic) {
        throw GenomeArchiveError("Not a genome archive: " + path.string());
    }
    if (header.version != genomeArchiveVersion) {
        throw GenomeArchiveError("Unsupported genome archive version " +
                                 std::to_string(header.version) + " in " + path.string());
    }
    if (header.weightCount != neuro::Topology::weightCount) {
        throw GenomeArchiveError("Genome archive stores " + std::to_string(header.weightCount) +
                                 " weights per genome, this build expects " +
                                 std::to_string(neuro::Topology::weightCount));
    }
    if (header.genomeCount == 0) {
        throw GenomeArchiveError("Genome archive contains no genomes: " + path.string());
    }

    GenomeArchive archive;
    archive.metadata = {header.generation,       header.scenario,        header.seed,
                        header.bestFitness,      header.meanFitness,     header.brainInputCount,
                        header.brainHiddenCount, header.brainOutputCount};
    archive.genomes.resize(header.genomeCount);
    for (Genome& genome : archive.genomes) {
        stream.read(reinterpret_cast<char*>(genome.weights.data()),
                    static_cast<std::streamsize>(genome.weights.size() * sizeof(float)));
        if (!stream || stream.gcount() !=
                           static_cast<std::streamsize>(genome.weights.size() * sizeof(float))) {
            throw GenomeArchiveError("Genome archive is truncated: " + path.string());
        }
    }
    return archive;
}

} // namespace vkexp
