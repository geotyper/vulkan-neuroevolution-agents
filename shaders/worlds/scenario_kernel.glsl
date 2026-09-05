#ifndef VKEXP_SCENARIO_KERNEL_GLSL
#define VKEXP_SCENARIO_KERNEL_GLSL

// vec2, uint, cos, sin, floor, max and length are built in here, so the shared
// kernel needs no shim on this side. See ScenarioKernel.inl for the rules the
// shared source follows.
#define VKEXP_KERNEL_FN
#include "vkexp/worlds/ScenarioKernel.inl"

#endif
