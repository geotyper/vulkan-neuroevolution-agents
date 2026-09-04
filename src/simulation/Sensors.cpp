#include "vkexp/simulation/Sensors.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace vkexp {
namespace {

float component(const Float4& value, const std::size_t index) {
    switch (index) {
    case 0:
        return value.x;
    case 1:
        return value.y;
    case 2:
        return value.z;
    default:
        return value.w;
    }
}

Float4 beaconColor(const std::size_t trial) {
    constexpr std::array colors{
        Float4{0.20F, 0.85F, 1.00F, 0.0F}, Float4{1.00F, 0.35F, 0.75F, 0.0F},
        Float4{0.55F, 1.00F, 0.35F, 0.0F}, Float4{1.00F, 0.72F, 0.20F, 0.0F}};
    return colors[trial % colors.size()];
}

float rangeFalloff(const float distance, const float range) {
    const float fade = std::clamp((range - distance) / (range * 0.25F), 0.0F, 1.0F);
    return fade * fade * (3.0F - 2.0F * fade);
}

} // namespace

neuro::Inputs sampleAgentInputs(const AgentState& agent, const SimulationStep& settings) {
    neuro::Inputs inputs{};
    const float dx = agent.target.x - agent.pose.x;
    const float dy = agent.target.y - agent.pose.y;
    const float distanceSquared = dx * dx + dy * dy;
    const float distance = std::sqrt(std::max(distanceSquared, 1.0e-8F));
    const float lightX = dx / distance;
    const float lightY = dy / distance;
    const float normalizedDistanceSquared =
        distanceSquared / (settings.worldRadius * settings.worldRadius);
    const float attenuation = 1.0F / (1.0F + normalizedDistanceSquared * 2.0F);
    const float falloff = rangeFalloff(distance, settings.lightSensorRange);
    const Float4 color = beaconColor(static_cast<std::size_t>(agent.target.z));

    for (std::size_t receptor = 0; receptor < neuro::Topology::lightReceptorCount; ++receptor) {
        const float fraction = static_cast<float>(receptor) /
                               static_cast<float>(neuro::Topology::lightReceptorCount - 1);
        const float sensorAngle = agent.pose.z + (fraction - 0.5F) * settings.sensorFieldOfView;
        const float alignment =
            std::max(std::cos(sensorAngle) * lightX + std::sin(sensorAngle) * lightY, 0.0F);
        const float intensity = std::pow(alignment, 12.0F) * attenuation * falloff;
        const std::size_t base = receptor * neuro::Topology::lightChannelsPerReceptor;
        inputs[base] = 1.0F - std::exp(-settings.lightExposure * color.x * intensity);
        inputs[base + 1] = 1.0F - std::exp(-settings.lightExposure * color.y * intensity);
        inputs[base + 2] = 1.0F - std::exp(-settings.lightExposure * color.z * intensity);
        inputs[base + 3] =
            inputs[base] * 0.2126F + inputs[base + 1] * 0.7152F + inputs[base + 2] * 0.0722F;
    }

    constexpr std::size_t tactileOffset =
        neuro::Topology::lightReceptorCount * neuro::Topology::lightChannelsPerReceptor;
    for (std::size_t sector = 0; sector < neuro::Topology::tactileSectorCount; ++sector) {
        const Float4& wall = sector < 4 ? agent.wallTouch0 : agent.wallTouch1;
        const Float4& other = sector < 4 ? agent.agentTouch0 : agent.agentTouch1;
        inputs[tactileOffset + sector * 2] = component(wall, sector % 4);
        inputs[tactileOffset + sector * 2 + 1] = component(other, sector % 4);
    }

    constexpr std::size_t selfOffset =
        tactileOffset +
        neuro::Topology::tactileSectorCount * neuro::Topology::tactileChannelsPerSector;
    const float speed =
        std::sqrt(agent.motion.x * agent.motion.x + agent.motion.y * agent.motion.y);
    inputs[selfOffset] = std::clamp(speed / settings.maximumSpeed, 0.0F, 1.0F);
    inputs[selfOffset + 1] = std::clamp(agent.motion.z / settings.maximumAngularSpeed, -1.0F, 1.0F);
    inputs[selfOffset + 2] = std::clamp(agent.motion.w, 0.0F, 1.0F);
    inputs[selfOffset + 3] = std::clamp(agent.signal.w, 0.0F, 1.0F);
    return inputs;
}

} // namespace vkexp
