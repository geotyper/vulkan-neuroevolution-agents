#include "vkexp/simulation/Sensors.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp {

neuro::Inputs samplePhotoreceptors(const AgentState& agent, const SimulationStep& settings) {
    neuro::Inputs inputs{};
    const float dx = agent.target.x - agent.pose.x;
    const float dy = agent.target.y - agent.pose.y;
    const float distanceSquared = dx * dx + dy * dy;
    const float inverseDistance = 1.0F / std::sqrt(std::max(distanceSquared, 1.0e-8F));
    const float lightX = dx * inverseDistance;
    const float lightY = dy * inverseDistance;
    const float worldRadiusSquared = settings.worldRadius * settings.worldRadius;
    const float normalizedDistanceSquared = distanceSquared / worldRadiusSquared;
    const float attenuation = 1.0F / (1.0F + normalizedDistanceSquared * 2.0F);

    for (std::size_t receptor = 0; receptor < neuro::Topology::receptorCount; ++receptor) {
        const float fraction =
            static_cast<float>(receptor) / static_cast<float>(neuro::Topology::receptorCount - 1);
        const float sensorAngle = agent.pose.z + (fraction - 0.5F) * settings.sensorFieldOfView;
        const float alignment =
            std::max(std::cos(sensorAngle) * lightX + std::sin(sensorAngle) * lightY, 0.0F);
        inputs[receptor] = std::pow(alignment, 12.0F) * attenuation;
    }
    const float speed =
        std::sqrt(agent.motion.x * agent.motion.x + agent.motion.y * agent.motion.y);
    inputs[7] = std::clamp(speed * 0.5F, 0.0F, 1.0F);
    inputs[8] = std::clamp(agent.motion.z / 3.0F, -1.0F, 1.0F);
    inputs[9] = std::clamp(agent.motion.w, 0.0F, 1.0F);
    return inputs;
}

} // namespace vkexp
