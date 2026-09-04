#version 450

struct Agent {
    vec4 pose;
    vec4 motion;
    vec4 signal;
    vec4 target;
    vec4 metrics;
    vec4 penalties;
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
    float beaconRotationAngle;
    float beaconRadiusRatio;
    float beaconMotionTime;
    float beaconTeleportProbability;
    uint beaconMotionSeed;
    uint reserved1;
    uint reserved2;
    uint reserved3;
} params;

layout(location = 0) out vec4 color;

const float Tau = 6.28318530718;

vec2 circleVertex(uint vertex, float radius, uint segmentCount) {
    const uint corner = vertex % 3;
    const uint segment = vertex / 3;
    if (corner == 0) {
        return vec2(0.0);
    }
    const float angle = Tau * float(segment + corner - 1) / float(segmentCount);
    return vec2(cos(angle), sin(angle)) * radius;
}

uint hashUint(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float random01(uint value) {
    return float(hashUint(value) & 0x00ffffffu) / 16777215.0;
}

vec2 randomWaypoint(uint waypoint, uint trial) {
    const uint key = params.beaconMotionSeed ^ (trial * 0x9e3779b9u) ^
                     (waypoint * 0x85ebca6bu);
    const float angle = random01(key) * Tau;
    const float radialFraction = 0.25 + sqrt(random01(key ^ 0xc2b2ae35u)) * 0.75;
    const float radius = params.worldRadius * params.beaconRadiusRatio * radialFraction;
    return vec2(cos(angle), sin(angle)) * radius;
}

vec2 randomMovingBeaconPosition(uint trial) {
    const float segmentValue = max(params.beaconMotionTime, 0.0) / 3.0;
    const uint segment = uint(floor(segmentValue));
    const float interpolation = fract(segmentValue);
    vec2 start = randomWaypoint(segment, trial);
    const uint eventKey = params.beaconMotionSeed ^ (trial * 0x27d4eb2du) ^
                          (segment * 0x165667b1u) ^ 0xa511e9b3u;
    if (random01(eventKey) < params.beaconTeleportProbability) {
        start = randomWaypoint(segment + 0x10000u, trial);
    }
    const vec2 end = randomWaypoint(segment + 1u, trial);
    const float smoothFactor = interpolation * interpolation * (3.0 - 2.0 * interpolation);
    return mix(start, end, smoothFactor);
}

void main() {
    Agent agent = agents[gl_InstanceIndex];
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
        bodyColor = mix(bodyColor, vec3(1.0, 0.35, 0.12), wallContact * 0.75);
        bodyColor = mix(bodyColor, vec3(1.0, 0.95, 0.35), agentContact * 0.85);
        color = vec4(bodyColor, params.opacity);
    } else if (params.mode == 1) {
        const vec3 trialColors[4] = vec3[](vec3(0.20, 0.85, 1.0), vec3(1.0, 0.35, 0.75),
                                           vec3(0.55, 1.0, 0.35), vec3(1.0, 0.72, 0.20));
        if (params.beaconScenario == 0) {
            world = agent.target.xy + circleVertex(gl_VertexIndex, 0.060, 16);
            color = vec4(trialColors[gl_InstanceIndex % 4], params.opacity);
        } else if (params.beaconScenario == 2) {
            const float orbitRadius = params.worldRadius * params.beaconRadiusRatio;
            const float targetLength = length(agent.target.xy);
            const vec2 base = targetLength > 0.000001
                                  ? agent.target.xy / targetLength * orbitRadius
                                  : vec2(orbitRadius, 0.0);
            const float cosine = cos(params.beaconRotationAngle);
            const float sine = sin(params.beaconRotationAngle);
            world = mat2(cosine, sine, -sine, cosine) * base;
            world += circleVertex(gl_VertexIndex, 0.060, 16);
            color = vec4(trialColors[gl_InstanceIndex % 4], params.opacity);
        } else if (params.beaconScenario == 3) {
            world = randomMovingBeaconPosition(gl_InstanceIndex);
            world += circleVertex(gl_VertexIndex, 0.060, 16);
            color = vec4(trialColors[gl_InstanceIndex % 4], params.opacity);
        } else {
            const float offset = params.worldRadius * 0.62;
            if (params.beaconPhase == 0) {
                world = gl_InstanceIndex == 0 ? vec2(-offset, -offset)
                                              : vec2(offset, offset);
            } else {
                world = gl_InstanceIndex == 0 ? vec2(-offset, offset)
                                              : vec2(offset, -offset);
            }
            world += circleVertex(gl_VertexIndex, 0.060, 16);
            color = vec4(trialColors[params.beaconPhase * 2 + gl_InstanceIndex],
                         params.opacity);
        }
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
