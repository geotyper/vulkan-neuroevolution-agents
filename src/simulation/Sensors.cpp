#include "vkexp/simulation/Sensors.hpp"

#include "vkexp/neuro/BrainKernel.hpp"

#include "vkexp/worlds/WorldScenario.hpp"

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

} // namespace

neuro::Inputs sampleAgentInputs(const AgentState& agent, const SimulationStep& settings) {
    neuro::Inputs inputs{};
    std::array<std::array<float, 3>, neuro::Topology::lightReceptorCount> radiance{};
    const ActiveBeacons beacons = activeBeacons(agent, settings);

    for (std::size_t beaconIndex = 0; beaconIndex < beacons.count; ++beaconIndex) {
        const Beacon& beacon = beacons.values[beaconIndex];
        // A wall that stops a body but not its light is a wall an agent can see
        // through, and the gradient then pulls it straight at the one place it
        // cannot go. Occlusion is per beacon, not per receptor: the beacon is a
        // point, so either the line to it is clear or the beacon is not there.
        if (sightBlocked(agent, settings, agent.pose.x, agent.pose.y, beacon.position.x,
                         beacon.position.y)) {
            continue;
        }
        const float dx = beacon.position.x - agent.pose.x;
        const float dy = beacon.position.y - agent.pose.y;
        const float distanceSquared = dx * dx + dy * dy;
        const float distance = std::sqrt(std::max(distanceSquared, 1.0e-8F));
        const float lightX = dx / distance;
        const float lightY = dy / distance;
        const float normalizedDistanceSquared =
            distanceSquared / (settings.worldRadius * settings.worldRadius);
        const float attenuation =
            neuro::kernel::brainDistanceAttenuation(normalizedDistanceSquared);
        const float falloff = neuro::kernel::brainRangeFalloff(distance, settings.lightSensorRange);
        for (std::size_t receptor = 0; receptor < neuro::Topology::lightReceptorCount; ++receptor) {
            const float fraction = static_cast<float>(receptor) /
                                   static_cast<float>(neuro::Topology::lightReceptorCount - 1);
            const float sensorAngle = agent.pose.z + (fraction - 0.5F) * settings.sensorFieldOfView;
            const float alignment =
                std::max(std::cos(sensorAngle) * lightX + std::sin(sensorAngle) * lightY, 0.0F);
            const float intensity =
                neuro::kernel::brainReceptorResponse(alignment) * attenuation * falloff;
            radiance[receptor][0] += beacon.color.x * intensity;
            radiance[receptor][1] += beacon.color.y * intensity;
            radiance[receptor][2] += beacon.color.z * intensity;
        }
    }

    // Every index below comes from the shared preset, so a new sensor block only
    // has to be declared once in BrainKernel.inl.
    namespace kernel = neuro::kernel;
    for (kernel::uint receptor = 0; receptor < kernel::BrainLightReceptorCount; ++receptor) {
        const float red = 1.0F - std::exp(-settings.lightExposure * radiance[receptor][0]);
        const float green = 1.0F - std::exp(-settings.lightExposure * radiance[receptor][1]);
        const float blue = 1.0F - std::exp(-settings.lightExposure * radiance[receptor][2]);
        inputs[kernel::brainLightChannelIndex(receptor, 0)] = red;
        inputs[kernel::brainLightChannelIndex(receptor, 1)] = green;
        inputs[kernel::brainLightChannelIndex(receptor, 2)] = blue;
        inputs[kernel::brainLightChannelIndex(receptor, 3)] =
            kernel::brainLuminance(red, green, blue);
    }

    for (kernel::uint sector = 0; sector < kernel::BrainTactileSectorCount; ++sector) {
        const Float4& wall = sector < 4 ? agent.wallTouch0 : agent.wallTouch1;
        const Float4& other = sector < 4 ? agent.agentTouch0 : agent.agentTouch1;
        inputs[kernel::brainTactileChannelIndex(sector, 0)] = component(wall, sector % 4);
        inputs[kernel::brainTactileChannelIndex(sector, 1)] = component(other, sector % 4);
    }

    const float speed =
        std::sqrt(agent.motion.x * agent.motion.x + agent.motion.y * agent.motion.y);
    inputs[kernel::BrainSelfOffset] = std::clamp(speed / settings.maximumSpeed, 0.0F, 1.0F);
    inputs[kernel::BrainSelfOffset + 1] =
        std::clamp(agent.motion.z / settings.maximumAngularSpeed, -1.0F, 1.0F);
    inputs[kernel::BrainSelfOffset + 2] = std::clamp(agent.motion.w, 0.0F, 1.0F);
    inputs[kernel::BrainSelfOffset + 3] = std::clamp(agent.signal.w, 0.0F, 1.0F);
    inputs[kernel::BrainTaskOffset] = std::clamp(agent.internal.x, 0.0F, 1.0F);
    inputs[kernel::BrainTaskOffset + 1] = std::clamp(agent.internal.y, 0.0F, 1.0F);
    inputs[kernel::BrainRecurrentInputOffset] = std::clamp(agent.internal.z, -1.0F, 1.0F);
    inputs[kernel::BrainRecurrentInputOffset + 1] = std::clamp(agent.internal.w, -1.0F, 1.0F);
    return inputs;
}

} // namespace vkexp
