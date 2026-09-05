#include "vkexp/worlds/scenarios/RotatingScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::rotating {
namespace {

float fitness(const AgentState& agent) {
    return objectiveFitness(agent, completedBeaconPhases(agent));
}

// floats0 = {rotation angle, orbit radius ratio, unused, unused}.
// Unpacked by rotatingScenarioPosition in shaders/worlds/rotating.glsl.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{settings.beaconRotationAngle, settings.beaconRadiusRatio, 0.0F, 0.0F}, {}, {}};
}

constexpr neuro::BrainShape brain{neuro::Topology::inputCount - neuro::Topology::taskInputCount -
                                      neuro::Topology::recurrentMemoryCount,
                                  neuro::Topology::hiddenCount,
                                  neuro::Topology::actuatorOutputCount};
static_assert(brain.fitsCapacity());

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{"Rotating", brain, fitness, gpuParameters};
    return value;
}

Float4 beaconPosition(const AgentState& agent, const SimulationStep& settings) {
    const float orbitRadius = settings.worldRadius * settings.beaconRadiusRatio;
    const float targetLength = std::hypot(agent.target.x, agent.target.y);
    const float baseX =
        targetLength > 0.000001F ? agent.target.x / targetLength * orbitRadius : orbitRadius;
    const float baseY =
        targetLength > 0.000001F ? agent.target.y / targetLength * orbitRadius : 0.0F;
    const float cosine = std::cos(settings.beaconRotationAngle);
    const float sine = std::sin(settings.beaconRotationAngle);
    return {baseX * cosine - baseY * sine, baseX * sine + baseY * cosine, 0.0F, 0.0F};
}

ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings) {
    const auto trial = static_cast<std::uint32_t>(std::max(agent.target.z, 0.0F));
    return {
        {{Beacon{beaconPosition(agent, settings), trialColors[trial % trialColors.size()]}, {}}},
        1};
}

float angleForStep(const float angularSpeed, const float deltaTime, const std::uint32_t step) {
    return std::fmod(angularSpeed * deltaTime * static_cast<float>(step), tau);
}

} // namespace vkexp::worlds::rotating
