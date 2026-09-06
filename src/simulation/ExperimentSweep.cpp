#include "vkexp/simulation/ExperimentSweep.hpp"

namespace vkexp {

void startSweep(SweepState& sweep) {
    sweep.stages.clear();
    sweep.stage = 0;
    sweep.generationsInStage = 0;
    sweep.running = false;
    if (sweep.values.empty() || sweep.generationsPerStage == 0) {
        return;
    }
    sweep.stages.push_back(SweepStage{sweep.values.front(), {}, {}, {}});
    sweep.running = true;
}

void stopSweep(SweepState& sweep) { sweep.running = false; }

float sweepValue(const SweepState& sweep) {
    if (sweep.stage >= sweep.values.size()) {
        return 0.0F;
    }
    return sweep.values[sweep.stage];
}

bool recordSweepGeneration(SweepState& sweep, const float bestFitness, const float medianFitness,
                           const float arrivalRatio) {
    if (!sweep.running || sweep.stages.empty()) {
        return false;
    }
    SweepStage& stage = sweep.stages.back();
    stage.bestFitness.push_back(bestFitness);
    stage.medianFitness.push_back(medianFitness);
    stage.arrivalRatio.push_back(arrivalRatio);
    if (++sweep.generationsInStage < sweep.generationsPerStage) {
        return false;
    }
    // The generation that fills a stage belongs to that stage and has already
    // been recorded above, so the restart the caller is about to perform costs
    // no measurement.
    sweep.generationsInStage = 0;
    if (++sweep.stage >= sweep.values.size()) {
        sweep.running = false;
        return false;
    }
    sweep.stages.push_back(SweepStage{sweep.values[sweep.stage], {}, {}, {}});
    return true;
}

} // namespace vkexp
