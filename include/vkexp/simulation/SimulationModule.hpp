#pragma once

#include "vkexp/core/Module.hpp"
#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/SimulationDriver.hpp"
#include "vkexp/simulation/SimulationState.hpp"

namespace vkexp {

class Profiler;

// Frame-loop adapter around SimulationDriver: it owns nothing of the simulation
// itself, only the mapping from module lifecycle to driver calls.
class SimulationModule final : public Module {
public:
    SimulationModule(SimulationState& state, Profiler& profiler, EvolutionSettings evolution = {},
                     SimulationDriverConfig config = {});

    void onAttach(AppContext& context) override;
    void onUpdate(AppContext& context, const FrameInfo& frame) override;
    void onRender(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

    [[nodiscard]] SimulationDriver& driver() { return driver_; }

private:
    SimulationState& state_;
    SimulationDriver driver_;
    ProfileMetricId metric_{invalidProfileMetric};
    bool finishPending_{};
};

} // namespace vkexp
