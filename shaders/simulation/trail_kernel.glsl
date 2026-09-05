#ifndef VKEXP_TRAIL_KERNEL_GLSL
#define VKEXP_TRAIL_KERNEL_GLSL

// vec2, uint and exp are built in here, so the shared kernel needs no shim on
// this side. See TrailKernel.inl for the rules the shared source follows.
#define VKEXP_TRAIL_FN
#include "vkexp/simulation/TrailKernel.inl"

#endif
