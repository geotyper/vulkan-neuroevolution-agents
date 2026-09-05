#ifndef VKEXP_SCENARIO_PARAMS_GLSL
#define VKEXP_SCENARIO_PARAMS_GLSL

// Opaque per-scenario transport block. Each scenario owns the meaning of these
// slots and unpacks them itself, so adding a scenario never widens the shared
// step parameters. Mirrors ScenarioParameterBlock in
// include/vkexp/simulation/AgentTypes.hpp (48 bytes, std430).
struct ScenarioParameters {
    vec4 floats0;
    vec4 floats1;
    uvec4 integers;
};

#endif
