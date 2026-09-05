#include "vkexp/simulation/CpuSimulation.hpp"

#include "vkexp/simulation/Sensors.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp {
namespace {

constexpr float foragePickupReward = 0.25F;
constexpr float forageDeliveryReward = 4.0F;

void setComponent(Float4& value, const std::size_t index, const float contact) {
    float* components[] = {&value.x, &value.y, &value.z, &value.w};
    *components[index] = std::max(*components[index], contact);
}

void recordWallContact(AgentState& agent, const float directionX, const float directionY,
                       const float strength) {
    constexpr float tau = 6.28318530718F;
    constexpr float halfSector = tau / 16.0F;
    float relative = std::fmod(std::atan2(directionY, directionX) - agent.pose.z + halfSector, tau);
    if (relative < 0.0F) {
        relative += tau;
    }
    const std::size_t sector = static_cast<std::size_t>(relative / tau * 8.0F) % 8;
    Float4& contacts = sector < 4 ? agent.wallTouch0 : agent.wallTouch1;
    setComponent(contacts, sector % 4, std::clamp(strength, 0.0F, 1.0F));
}

} // namespace

void stepAgentCpu(AgentState& agent,
                  const std::span<const float, neuro::Topology::weightCount> weights,
                  const SimulationStep& settings) {
    if (settings.beaconScenario == BeaconScenario::AlternatingDiagonals &&
        settings.beaconPhaseChanged) {
        agent.metrics.w += agent.metrics.x - agent.metrics.y;
        agent.metrics.x = nearestBeaconDistance(agent, settings);
        agent.metrics.y = agent.metrics.x;
    }
    if (settings.beaconScenario == BeaconScenario::ForageHome && agent.internal.y >= 0.5F &&
        homeBeaconRelocated(settings)) {
        agent.metrics.w += std::max(agent.metrics.x - agent.metrics.y, 0.0F);
        agent.metrics.x = nearestBeaconDistance(agent, settings);
        agent.metrics.y = agent.metrics.x;
    }
    const neuro::BrainShape brain = scenarioDefinition(settings.beaconScenario).brain;
    const neuro::Outputs output =
        neuro::evaluate(weights, sampleAgentInputs(agent, settings), brain);
    const float left = output[0];
    const float right = output[1];
    const float forwardX = std::cos(agent.pose.z);
    const float forwardY = std::sin(agent.pose.z);
    const float drive = 0.5F * (left + right) * settings.thrust;

    agent.motion.x += forwardX * drive * settings.deltaTime;
    agent.motion.y += forwardY * drive * settings.deltaTime;
    const float linearDamping = std::exp(-settings.linearDrag * settings.deltaTime);
    agent.motion.x *= linearDamping;
    agent.motion.y *= linearDamping;
    agent.motion.z += (right - left) * settings.turnAcceleration * settings.deltaTime;
    agent.motion.z *= std::exp(-settings.angularDrag * settings.deltaTime);
    const float speed =
        std::sqrt(agent.motion.x * agent.motion.x + agent.motion.y * agent.motion.y);
    if (speed > settings.maximumSpeed) {
        const float scale = settings.maximumSpeed / speed;
        agent.motion.x *= scale;
        agent.motion.y *= scale;
    }
    agent.motion.z =
        std::clamp(agent.motion.z, -settings.maximumAngularSpeed, settings.maximumAngularSpeed);
    agent.pose.x += agent.motion.x * settings.deltaTime;
    agent.pose.y += agent.motion.y * settings.deltaTime;
    agent.pose.z += agent.motion.z * settings.deltaTime;
    agent.wallTouch0 = {};
    agent.wallTouch1 = {};
    agent.agentTouch0 = {};
    agent.agentTouch1 = {};

    const float maximumCenterDistance = std::max(settings.worldRadius - agent.pose.w, 0.0F);
    if (settings.worldShape == WorldShape::Circle) {
        const float centerDistance =
            std::sqrt(agent.pose.x * agent.pose.x + agent.pose.y * agent.pose.y);
        if (centerDistance > maximumCenterDistance) {
            const float normalX = agent.pose.x / centerDistance;
            const float normalY = agent.pose.y / centerDistance;
            agent.pose.x = normalX * maximumCenterDistance;
            agent.pose.y = normalY * maximumCenterDistance;
            const float outward = agent.motion.x * normalX + agent.motion.y * normalY;
            if (outward > 0.0F) {
                agent.motion.x -= normalX * outward * 1.5F;
                agent.motion.y -= normalY * outward * 1.5F;
            }
            recordWallContact(agent, normalX, normalY,
                              0.25F + std::max(outward, 0.0F) / settings.maximumSpeed);
        }
    } else {
        if (agent.pose.x > maximumCenterDistance) {
            agent.pose.x = maximumCenterDistance;
            recordWallContact(agent, 1.0F, 0.0F,
                              0.25F + std::max(agent.motion.x, 0.0F) / settings.maximumSpeed);
            agent.motion.x = std::min(agent.motion.x, 0.0F) - std::max(agent.motion.x, 0.0F) * 0.5F;
        } else if (agent.pose.x < -maximumCenterDistance) {
            agent.pose.x = -maximumCenterDistance;
            recordWallContact(agent, -1.0F, 0.0F,
                              0.25F - std::min(agent.motion.x, 0.0F) / settings.maximumSpeed);
            agent.motion.x = std::max(agent.motion.x, 0.0F) - std::min(agent.motion.x, 0.0F) * 0.5F;
        }
        if (agent.pose.y > maximumCenterDistance) {
            agent.pose.y = maximumCenterDistance;
            recordWallContact(agent, 0.0F, 1.0F,
                              0.25F + std::max(agent.motion.y, 0.0F) / settings.maximumSpeed);
            agent.motion.y = std::min(agent.motion.y, 0.0F) - std::max(agent.motion.y, 0.0F) * 0.5F;
        } else if (agent.pose.y < -maximumCenterDistance) {
            agent.pose.y = -maximumCenterDistance;
            recordWallContact(agent, 0.0F, -1.0F,
                              0.25F - std::min(agent.motion.y, 0.0F) / settings.maximumSpeed);
            agent.motion.y = std::max(agent.motion.y, 0.0F) - std::min(agent.motion.y, 0.0F) * 0.5F;
        }
    }
    const float wallContactAmount = agent.wallTouch0.x + agent.wallTouch0.y + agent.wallTouch0.z +
                                    agent.wallTouch0.w + agent.wallTouch1.x + agent.wallTouch1.y +
                                    agent.wallTouch1.z + agent.wallTouch1.w;
    agent.penalties.x += wallContactAmount * settings.wallCollisionPenalty;

    const float motorCost = (std::abs(left) + std::abs(right)) * settings.deltaTime;
    const float signalIntensity = std::max(output[5], 0.0F);
    const float signalCost = signalIntensity * settings.deltaTime * 0.25F;
    agent.motion.w = std::max(0.0F, agent.motion.w - motorCost * 0.0008F - signalCost * 0.0008F);
    agent.signal = {output[2] * 0.5F + 0.5F, output[3] * 0.5F + 0.5F, output[4] * 0.5F + 0.5F,
                    signalIntensity};
    if (brain.outputCount >=
        neuro::Topology::actuatorOutputCount + neuro::Topology::recurrentMemoryCount) {
        agent.internal.z = output[6];
        agent.internal.w = output[7];
    } else {
        agent.internal.z = 0.0F;
        agent.internal.w = 0.0F;
    }

    const float distance = nearestBeaconDistance(agent, settings);
    agent.metrics.y = std::min(agent.metrics.y, distance);
    agent.metrics.z += motorCost + signalCost;
    if (settings.beaconScenario == BeaconScenario::ForageHome) {
        if (agent.internal.y >= 0.5F) {
            agent.internal.x = std::max(0.0F, agent.internal.x - settings.forageCargoDecayRate *
                                                                     settings.deltaTime);
        }
        if (distance < beaconArrivalRadius(settings)) {
            agent.metrics.w += std::max(agent.metrics.x - agent.metrics.y, 0.0F);
            if (agent.internal.y >= 0.5F) {
                agent.metrics.w += agent.internal.x * forageDeliveryReward;
                agent.target.w = static_cast<float>(completedForageCycles(agent) + 1);
                agent.internal.x = 0.0F;
                agent.internal.y = 0.0F;
            } else {
                agent.metrics.w += foragePickupReward;
                agent.internal.x = 1.0F;
                agent.internal.y = 1.0F;
            }
            agent.metrics.x = nearestBeaconDistance(agent, settings);
            agent.metrics.y = agent.metrics.x;
        }
        return;
    }
    if (settings.beaconScenario == BeaconScenario::Rotating ||
        settings.beaconScenario == BeaconScenario::RandomMovement) {
        const float visibleCloseness =
            std::clamp(1.0F - distance / settings.lightSensorRange, 0.0F, 1.0F);
        agent.metrics.w += visibleCloseness * settings.deltaTime * 0.25F;
    }
    if (distance < beaconArrivalRadius(settings)) {
        const std::uint32_t phaseBit = 1U << settings.beaconPhase;
        const auto completedMask =
            static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F) + 0.5F) | phaseBit;
        agent.target.w = static_cast<float>(completedMask);
    }
}

float agentFitness(const AgentState& agent, const BeaconScenario scenario) {
    return scenarioDefinition(scenario).fitness(agent);
}

} // namespace vkexp
