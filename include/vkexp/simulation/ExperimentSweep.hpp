#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vkexp {

// Enough stages to bracket a setting and see the shape between the ends, and
// few enough that every stage still gets a legible plot of its own.
inline constexpr std::size_t maximumSweepStages = 6;

// One leg of a sweep: the setting it ran at, and the curves it produced.
struct SweepStage {
    float groupSharing{};
    std::vector<float> bestFitness;
    std::vector<float> medianFitness;
    std::vector<float> arrivalRatio;
};

// Several runs of one experiment that differ in a single setting, kept side by
// side. The comparison is the point, so the stages are driven from here rather
// than from a shell loop: they share a seed, a scenario and an arena, and the
// only thing that moves between them is the swept value.
//
// Plain data plus the three functions that advance it. Nothing here needs a
// device, which is why the stage boundary -- the part with an off-by-one in it
// -- is decided by a unit test rather than by watching the window.
struct SweepState {
    // The plan. Editable while stopped; `startSweep` reads it as it stands.
    std::vector<float> values{0.0F, 0.5F, 1.0F};
    std::uint32_t generationsPerStage{60};

    bool running{};
    std::size_t stage{};
    std::uint32_t generationsInStage{};

    // Completed stages, plus the one in flight while running. Kept after the
    // sweep ends -- results outliving the run is the only reason to have them.
    std::vector<SweepStage> stages;
};

// Clears any previous result and arms the first stage of the plan already in
// `sweep`. A plan with no values or no generations per stage does not start, so
// a caller never has to handle a sweep that runs but has nothing to run.
void startSweep(SweepState& sweep);

// Stops without discarding what has been measured so far.
void stopSweep(SweepState& sweep);

// The value the current stage runs at, and 0 once the plan is exhausted.
[[nodiscard]] float sweepValue(const SweepState& sweep);

// Records one finished generation into the stage in flight. Returns true when
// that stage just filled up and the next one is armed -- the caller's cue to
// apply `sweepValue` and restart evolution. The final stage returns false and
// leaves the sweep stopped with every stage kept.
[[nodiscard]] bool recordSweepGeneration(SweepState& sweep, float bestFitness, float medianFitness,
                                         float arrivalRatio);

} // namespace vkexp
