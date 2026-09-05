#pragma once

#include <cstdint>

// The simulation has always run in a coherent metric scale -- a body 4.4 cm
// across in an arena 3.7 m wide, moving at up to 0.55 m/s -- but nothing said
// so, which left every constant reading as an arbitrary number and let a new
// one be written at the wrong scale without complaint. This header is where
// the scale is declared and where the step's two time bases meet.
//
// Steps and seconds both stay, with different jobs:
//   * a step is the unit of reproducibility. A run is replayed by step count,
//     genome archives record steps, and the CPU/GPU parity tests compare step
//     for step. Nothing about a replay depends on wall-clock time.
//   * a second is the unit every physical quantity is expressed in. Speeds are
//     m/s, accelerations m/s^2, drags and decay rates 1/s, and anything the
//     simulation accumulates over time is scaled by `deltaTime`.
//
// The rule that keeps the two consistent: a quantity accumulated by the step
// is multiplied by `deltaTime`, and a fraction removed per step is written as
// `1 - exp(-rate * deltaTime)`. Halving `deltaTime` must then leave behaviour
// unchanged, which `testFixedStepIndependence` checks.
namespace vkexp::units {

// One world unit is one metre. Every length below follows from that choice.
inline constexpr float metresPerUnit = 1.0F;

// Integration rate. Sensors, physics and the brain all advance together at
// this rate; there is no substepping.
inline constexpr float simulationRateHz = 60.0F;
inline constexpr float fixedTimeStep = 1.0F / simulationRateHz;

[[nodiscard]] constexpr float secondsForSteps(const std::uint32_t steps, const float deltaTime) {
    return static_cast<float>(steps) * deltaTime;
}

[[nodiscard]] constexpr float metresToCentimetres(const float metres) { return metres * 100.0F; }

} // namespace vkexp::units
