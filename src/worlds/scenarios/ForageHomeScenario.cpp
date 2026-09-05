#include "vkexp/worlds/scenarios/ForageHomeScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"
#include "vkexp/worlds/scenarios/RotatingScenario.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::forage_home {
namespace {

constexpr Float4 resourceColor{1.00F, 0.55F, 0.08F, 0.0F};
constexpr Float4 homeColor{0.12F, 0.72F, 1.00F, 0.0F};
constexpr float minimumHomeRadiusRatio = 0.38F;
constexpr float homeRadiusRange = 0.32F;

} // namespace

Float4 homePosition(const AgentState& agent, const SimulationStep& settings) {
    const auto trial = static_cast<std::uint32_t>(std::max(agent.target.z, 0.0F));
    const auto epoch = static_cast<std::uint32_t>(
        std::floor(std::max(settings.beaconMotionTime, 0.0F) / forageHomeRelocationSeconds));
    const std::uint32_t key =
        settings.beaconMotionSeed ^ (trial * 0x51ed270bU) ^ (epoch * 0x85ebca6bU) ^ 0xc2b2ae35U;
    const float angle = random01(key) * tau;
    const float radiusRatio = minimumHomeRadiusRatio + random01(key ^ 0x27d4eb2dU) * homeRadiusRange;
    const float radius = settings.worldRadius * radiusRatio;
    return {std::cos(angle) * radius, std::sin(angle) * radius, 0.0F, 0.0F};
}

bool homeRelocated(const SimulationStep& settings) {
    const float currentTime = std::max(settings.beaconMotionTime, 0.0F);
    const float previousTime = std::max(currentTime - settings.deltaTime, 0.0F);
    const float epsilon = std::max(settings.deltaTime * 0.01F, 0.000001F);
    return static_cast<std::uint32_t>(
               std::floor((currentTime + epsilon) / forageHomeRelocationSeconds)) !=
           static_cast<std::uint32_t>(
               std::floor((previousTime + epsilon) / forageHomeRelocationSeconds));
}

ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings) {
    return {{{Beacon{rotating::beaconPosition(agent, settings), resourceColor},
              Beacon{homePosition(agent, settings), homeColor}}},
            2};
}

float targetDistance(const AgentState& agent, const SimulationStep& settings) {
    const ActiveBeacons active = beacons(agent, settings);
    const std::size_t targetIndex = agent.internal.y >= 0.5F ? 1 : 0;
    const float dx = active.values[targetIndex].position.x - agent.pose.x;
    const float dy = active.values[targetIndex].position.y - agent.pose.y;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace vkexp::worlds::forage_home
