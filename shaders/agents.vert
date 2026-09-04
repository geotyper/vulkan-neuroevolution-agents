#version 450

struct Agent {
    vec4 pose;
    vec4 motion;
    vec4 signal;
    vec4 target;
    vec4 metrics;
};

layout(std430, set = 0, binding = 0) readonly buffer Agents {
    Agent agents[];
};

layout(push_constant) uniform DrawParameters {
    float scaleX;
    float scaleY;
    float worldRadius;
    float opacity;
    uint mode; // 0 hex body, 1 beacon, 2 light halo, 3 arena, 4 heading
    uint worldShape;
    uint reserved1;
    uint reserved2;
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

void main() {
    Agent agent = agents[gl_InstanceIndex];
    vec2 world;
    if (params.mode == 0) {
        world = agent.pose.xy + circleVertex(gl_VertexIndex, agent.pose.w, 16);
        color = vec4(mix(vec3(0.72, 0.82, 0.92), agent.signal.rgb, 0.42), params.opacity);
    } else if (params.mode == 1) {
        world = agent.target.xy + circleVertex(gl_VertexIndex, 0.060, 16);
        const vec3 trialColors[4] = vec3[](vec3(0.20, 0.85, 1.0), vec3(1.0, 0.35, 0.75),
                                           vec3(0.55, 1.0, 0.35), vec3(1.0, 0.72, 0.20));
        color = vec4(trialColors[gl_InstanceIndex % 4], params.opacity);
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
        color = vec4(0.055, 0.09, 0.14, params.opacity);
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
