#include "vkexp/simulation/CpuSimulation.hpp"

#include "vkexp/simulation/Sensors.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp {

void stepAgentCpu(AgentState& agent,
                  const std::span<const float, neuro::Topology::weightCount> weights,
                  const SimulationStep& settings) {
    const neuro::Outputs output = neuro::evaluate(weights, samplePhotoreceptors(agent, settings));
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
        }
    } else {
        if (agent.pose.x > maximumCenterDistance) {
            agent.pose.x = maximumCenterDistance;
            agent.motion.x = std::min(agent.motion.x, 0.0F) - std::max(agent.motion.x, 0.0F) * 0.5F;
        } else if (agent.pose.x < -maximumCenterDistance) {
            agent.pose.x = -maximumCenterDistance;
            agent.motion.x = std::max(agent.motion.x, 0.0F) - std::min(agent.motion.x, 0.0F) * 0.5F;
        }
        if (agent.pose.y > maximumCenterDistance) {
            agent.pose.y = maximumCenterDistance;
            agent.motion.y = std::min(agent.motion.y, 0.0F) - std::max(agent.motion.y, 0.0F) * 0.5F;
        } else if (agent.pose.y < -maximumCenterDistance) {
            agent.pose.y = -maximumCenterDistance;
            agent.motion.y = std::max(agent.motion.y, 0.0F) - std::min(agent.motion.y, 0.0F) * 0.5F;
        }
    }

    const float motorCost = (std::abs(left) + std::abs(right)) * settings.deltaTime;
    agent.motion.w = std::max(0.0F, agent.motion.w - motorCost * 0.0008F);
    agent.signal = {output[2] * 0.5F + 0.5F, output[3] * 0.5F + 0.5F, output[4] * 0.5F + 0.5F,
                    std::max(output[5], 0.0F)};

    const float targetX = agent.target.x - agent.pose.x;
    const float targetY = agent.target.y - agent.pose.y;
    const float distance = std::sqrt(targetX * targetX + targetY * targetY);
    agent.metrics.y = std::min(agent.metrics.y, distance);
    agent.metrics.z += motorCost;
    if (distance < settings.arrivalRadius) {
        agent.metrics.w = 1.0F;
    }
}

float agentFitness(const AgentState& agent) {
    return (agent.metrics.x - agent.metrics.y) + agent.metrics.w * 2.0F - agent.metrics.z * 0.002F;
}

} // namespace vkexp
