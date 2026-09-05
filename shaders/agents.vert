#version 450
#extension GL_GOOGLE_include_directive : require

#include "worlds/scenario_params.glsl"

struct Agent {
    vec4 pose;
    vec4 motion;
    vec4 signal;
    vec4 target;
    vec4 metrics;
    vec4 penalties;
    vec4 internal;
    vec4 wallTouch0;
    vec4 wallTouch1;
    vec4 agentTouch0;
    vec4 agentTouch1;
};

layout(std430, set = 0, binding = 0) readonly buffer Agents {
    Agent agents[];
};

layout(push_constant) uniform DrawParameters {
    float scaleX;
    float scaleY;
    float worldRadius;
    float opacity;
    uint mode; // 0 circular body, 1 beacon, 2 light halo, 3 arena, 4 heading
    uint worldShape;
    uint beaconScenario;
    uint beaconPhase;
    uint selectedWorld;
    uint agentsPerWorld;
    uint trialsPerGenome;
    uint reserved;
    ScenarioParameters scenario;
} params;

#include "worlds/world_scenarios.glsl"

layout(location = 0) out vec4 color;

const float BeaconRadius = 0.060;

vec2 circleVertex(uint vertex, float radius, uint segmentCount) {
    const uint corner = vertex % 3;
    const uint segment = vertex / 3;
    if (corner == 0) {
        return vec2(0.0);
    }
    const float angle = ScenarioTau * float(segment + corner - 1) / float(segmentCount);
    return vec2(cos(angle), sin(angle)) * radius;
}

void main() {
    const uint trial = params.selectedWorld % params.trialsPerGenome;
    const uint group = params.selectedWorld / params.trialsPerGenome;
    const uint firstGenome = group * params.agentsPerWorld;
    const uint agentIndex =
        (firstGenome + gl_InstanceIndex) * params.trialsPerGenome + trial;
    const uint firstAgentIndex = firstGenome * params.trialsPerGenome + trial;
    Agent agent = agents[params.mode == 0u || params.mode == 2u || params.mode == 4u
                             ? agentIndex
                             : firstAgentIndex];
    vec2 world;
    if (params.mode == 0) {
        world = agent.pose.xy + circleVertex(gl_VertexIndex, agent.pose.w, 16);
        const float wallContact = max(max(max(agent.wallTouch0.x, agent.wallTouch0.y),
                                          max(agent.wallTouch0.z, agent.wallTouch0.w)),
                                      max(max(agent.wallTouch1.x, agent.wallTouch1.y),
                                          max(agent.wallTouch1.z, agent.wallTouch1.w)));
        const float agentContact = max(max(max(agent.agentTouch0.x, agent.agentTouch0.y),
                                           max(agent.agentTouch0.z, agent.agentTouch0.w)),
                                       max(max(agent.agentTouch1.x, agent.agentTouch1.y),
                                           max(agent.agentTouch1.z, agent.agentTouch1.w)));
        vec3 bodyColor = mix(vec3(0.72, 0.82, 0.92), agent.signal.rgb, 0.42);
        bodyColor = scenarioBodyTint(params.beaconScenario, agent, bodyColor);
        bodyColor = mix(bodyColor, vec3(1.0, 0.35, 0.12), wallContact * 0.75);
        bodyColor = mix(bodyColor, vec3(1.0, 0.95, 0.35), agentContact * 0.85);
        color = vec4(bodyColor, params.opacity);
    } else if (params.mode == 1) {
        world = scenarioBeaconPosition(params.beaconScenario, agent, gl_InstanceIndex,
                                       params.beaconPhase, params.worldRadius, params.scenario) +
                circleVertex(gl_VertexIndex, BeaconRadius, 16);
        color = vec4(scenarioBeaconColor(params.beaconScenario, gl_InstanceIndex,
                                         params.beaconPhase, trial),
                     params.opacity);
    } else if (params.mode == 2) {
        const float radius = agent.pose.w + agent.signal.a * 0.055;
        world = agent.pose.xy + circleVertex(gl_VertexIndex, radius, 16);
        color = vec4(agent.signal.rgb, params.opacity * agent.signal.a);
    } else if (params.mode == 3) {
        if (params.worldShape == 0) {
            world = circleVertex(gl_VertexIndex, params.worldRadius, 64);
        } else {
            const float r = params.worldRadius;
            const vec2 square[6] = vec2[](vec2(-r, -r), vec2(r, -r), vec2(r, r),
                                          vec2(-r, -r), vec2(r, r), vec2(-r, r));
            world = square[gl_VertexIndex];
        }
        color = vec4(0.022, 0.034, 0.055, params.opacity);
    } else {
        const vec2 heading[3] = vec2[](vec2(1.42, 0.0), vec2(0.25, 0.38),
                                       vec2(0.25, -0.38));
        const vec2 local = heading[gl_VertexIndex] * agent.pose.w;
        const float c = cos(agent.pose.z);
        const float s = sin(agent.pose.z);
        world = agent.pose.xy + mat2(c, s, -s, c) * local;
        color = vec4(0.92, 0.97, 1.0, params.opacity);
    }
    gl_Position = vec4(world.x * params.scaleX, world.y * params.scaleY, 0.0, 1.0);
}
