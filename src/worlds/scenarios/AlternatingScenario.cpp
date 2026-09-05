#include "vkexp/worlds/scenarios/AlternatingScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

namespace vkexp::worlds::alternating {
namespace {

float fitness(const AgentState& agent) {
    return objectiveFitness(agent, completedBeaconPhases(agent));
}

constexpr neuro::BrainShape brain{neuro::Topology::inputCount - neuro::Topology::taskInputCount -
                                      neuro::Topology::recurrentMemoryCount,
                                  neuro::Topology::hiddenCount,
                                  neuro::Topology::actuatorOutputCount};
static_assert(brain.fitsCapacity());

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{"Alternating diagonals", brain, fitness};
    return value;
}

ActiveBeacons beacons(const SimulationStep& settings) {
    const float offset = settings.worldRadius * 0.62F;
    const bool secondDiagonal = settings.beaconPhase != 0;
    const float firstY = secondDiagonal ? offset : -offset;
    const float secondY = -firstY;
    const std::size_t colorOffset = secondDiagonal ? 2 : 0;
    return {{{Beacon{{-offset, firstY, 0.0F, 0.0F}, trialColors[colorOffset]},
              Beacon{{offset, secondY, 0.0F, 0.0F}, trialColors[colorOffset + 1]}}},
            2};
}

std::uint32_t phaseForStep(const std::uint32_t step, const std::uint32_t stepsPerGeneration) {
    if (stepsPerGeneration == 0) {
        return 0;
    }
    return step >= stepsPerGeneration / 2 ? 1U : 0U;
}

} // namespace vkexp::worlds::alternating
