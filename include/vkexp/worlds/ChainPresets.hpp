#pragma once

#include "vkexp/simulation/AgentTypes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

// Named formations for the chain world.
//
// The same arrangement as Locomotion.hpp and for the same reason: the sliders
// already reach everywhere, so what a preset adds is not reach but a claim --
// this combination is a thing, it has a name, and here is what it does. A point
// in the space with a story, rather than nine numbers to be rediscovered.
//
// Unlike the locomotion table these touch three settings that are not the chain
// world's own: the speed floor, the wall penalty and group fitness sharing. That
// is deliberate and it is the whole point. Without a floor the best answer is to
// stop in a good arrangement; without a wall penalty the arena rim is a free
// place to park, because a floor holds the velocity while the contact cancels
// the displacement; and without sharing, the ends of a chain are punished
// individually and defect. A preset that set only the chain numbers would be a
// preset that does not work, which is exactly the trap these exist to close.
//
// -- What separates them --
//
// Read the table by which knob moves. Column and Mill differ in one number, the
// heading alignment, and that difference is the finding this world was built to
// produce: an arrangement cannot tell a column from a ring, because at every
// instant both have their neighbours where the rule wants them. Only the motion
// can. Turn alignment on and the line travels along itself; turn it off and the
// same line is free to close into a rotating loop.
//
// That loop is not a failure. It is the ant mill and the fish torus, and it was
// what this world kept producing before there was a term that could see it. So
// it is in the table under its own name rather than described as a thing to
// avoid.
namespace vkexp::worlds {

enum class ChainFormation : std::uint32_t {
    Column = 0,
    Mill = 1,
    Flock = 2,
    Pairs = 3,
};

inline constexpr std::size_t chainFormationCount = 4;

struct ChainPreset {
    ChainFormation formation{};
    const char* name{};
    // Stable identifier for command lines; unlike `name` it must not change once
    // runs have been recorded against it.
    const char* key{};
    float neighbourBodies{};
    std::uint32_t rewardBand{};
    std::uint32_t crowdLimit{};
    float crowdPenalty{};
    float straightWeight{};
    float alignWeight{};
    // Not the chain world's own, and set anyway. See the note above.
    float minimumSpeed{};
    float wallPenalty{};
    float groupSharing{};
    const char* description{};
};

inline constexpr std::array<ChainPreset, chainFormationCount> chainPresets{{
    {ChainFormation::Column, "Column", "column", 1.5F, 2U, 3U, 1.0F, 1.0F, 0.7F, 0.20F, 3.0F, 1.0F,
     "A line travelling along itself. Neighbours ahead and behind, everyone going the same way, "
     "and a third neighbour costs -- three this close is already a triangle. The formation this "
     "world was built for, and the hardest of the four: a line has ends, and an end is worth less "
     "than a middle, so it only holds together when the world is scored as a whole."},
    {ChainFormation::Mill, "Mill", "mill", 1.5F, 2U, 2U, 2.0F, 1.0F, 0.0F, 0.30F, 3.0F, 1.0F,
     "The same line, allowed to close. One number apart from Column -- heading alignment off -- "
     "because that is the only thing that can tell a ring from a line: the arrangement is "
     "identical and the motion is not. A strict limit of two neighbours keeps the structure one "
     "agent wide, and the faster floor keeps it turning. This is the ant mill and the fish torus, "
     "and it is what this world produced for days before there was a term that could see it."},
    {ChainFormation::Flock, "Flock", "flock", 3.0F, 4U, 8U, 0.5F, 0.0F, 1.0F, 0.25F, 2.0F, 1.0F,
     "Go the same way; stand where you like. Straightness off and alignment at its maximum, with "
     "a wide radius and room for eight neighbours, so nothing is asked about shape at all. The "
     "control for both of the others: whatever a flock achieves here, Column and Mill have to "
     "beat before their arrangement terms can be said to have done anything."},
    {ChainFormation::Pairs, "Pairs", "pairs", 1.5F, 1U, 1U, 3.0F, 0.0F, 1.0F, 0.25F, 3.0F, 0.0F,
     "Exactly one partner, moving together, and a second is punished hard. The one preset scored "
     "individually rather than by world, because a pair has no ends: both halves are worth the "
     "same, so nothing has to be shared to keep it together. The cheapest formation here, and "
     "worth running first -- if the others cannot beat two agents holding hands, their extra "
     "structure is not paying for itself."},
}};

[[nodiscard]] inline const ChainPreset& chainPreset(const ChainFormation formation) {
    const auto index = static_cast<std::size_t>(formation);
    return chainPresets[std::min(index, chainFormationCount - 1)];
}

inline void applyChainPreset(SimulationStep& settings, const ChainFormation formation) {
    const ChainPreset& preset = chainPreset(formation);
    settings.chainNeighbourBodies = preset.neighbourBodies;
    settings.chainRewardBand = preset.rewardBand;
    settings.chainCrowdLimit = preset.crowdLimit;
    settings.chainCrowdPenalty = preset.crowdPenalty;
    settings.chainStraightWeight = preset.straightWeight;
    settings.chainAlignWeight = preset.alignWeight;
    settings.minimumSpeed = preset.minimumSpeed;
    settings.wallCollisionPenalty = preset.wallPenalty;
    settings.fitness.groupSharing = preset.groupSharing;
}

// Which preset the sliders are sitting on, if any. The window needs this because
// every value is also editable one at a time: a combo that could only be set and
// never read would go on saying "Mill" over a world that had since been dragged
// somewhere else.
[[nodiscard]] inline bool matchesChainPreset(const SimulationStep& settings,
                                             const ChainFormation formation) {
    const ChainPreset& preset = chainPreset(formation);
    const auto near = [](const float value, const float reference) {
        return std::abs(value - reference) <= 0.005F * std::max(std::abs(reference), 1.0F);
    };
    return settings.chainRewardBand == preset.rewardBand &&
           settings.chainCrowdLimit == preset.crowdLimit &&
           near(settings.chainNeighbourBodies, preset.neighbourBodies) &&
           near(settings.chainCrowdPenalty, preset.crowdPenalty) &&
           near(settings.chainStraightWeight, preset.straightWeight) &&
           near(settings.chainAlignWeight, preset.alignWeight) &&
           near(settings.minimumSpeed, preset.minimumSpeed) &&
           near(settings.wallCollisionPenalty, preset.wallPenalty) &&
           near(settings.fitness.groupSharing, preset.groupSharing);
}

[[nodiscard]] inline const ChainPreset* currentChainPreset(const SimulationStep& settings) {
    for (const ChainPreset& preset : chainPresets) {
        if (matchesChainPreset(settings, preset.formation)) {
            return &preset;
        }
    }
    return nullptr;
}

[[nodiscard]] inline const ChainPreset* chainPresetForKey(const std::string_view key) {
    for (const ChainPreset& preset : chainPresets) {
        if (key == preset.key) {
            return &preset;
        }
    }
    return nullptr;
}

} // namespace vkexp::worlds
