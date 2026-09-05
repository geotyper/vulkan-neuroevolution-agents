#ifndef VKEXP_SCENARIO_MATH_GLSL
#define VKEXP_SCENARIO_MATH_GLSL

#include "worlds/scenario_kernel.glsl"

// Kept on this side only because the CPU stores trial colours as Float4s while
// the shader wants a vec3; the values themselves live in one place per language
// and are compared by the CPU/GPU parity tests.
vec3 trialPaletteColor(uint index) {
    const vec3 colors[4] = vec3[](vec3(0.20, 0.85, 1.00), vec3(1.00, 0.35, 0.75),
                                  vec3(0.55, 1.00, 0.35), vec3(1.00, 0.72, 0.20));
    return colors[index % 4];
}

#endif
