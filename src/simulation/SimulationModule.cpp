#include "vkexp/simulation/SimulationModule.hpp"

#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/profiling/Profiler.hpp"

#include <exception>

namespace vkexp {

// The driver writes per-step parameters into host-visible memory during
// onRender. That is only safe because VulkanContext waits on the frame fence in
// beginFrame, so no earlier submission can still be reading them.
static_assert(VulkanContext::framesInFlight == 1,
              "SimulationDriver's host-visible step parameters need per-frame slots "
              "once more than one frame is in flight");

SimulationModule::SimulationModule(SimulationState& state, Profiler& profiler,
                                   EvolutionSettings evolution, SimulationDriverConfig config)
    : state_(state), driver_(state, evolution, config),
      metric_(profiler.registerMetric("Agent simulation")) {}

void SimulationModule::onAttach(AppContext& context) {
    driver_.createResources(context.vulkan.physicalDevice(), context.vulkan.device());
}

void SimulationModule::onUpdate(AppContext& context, const FrameInfo&) {
    // Before the reset and step branches: a snapshot is about the run as it
    // stands, so it must not race a restart or a pending generation boundary.
    if (state_.controls.saveRequested || state_.controls.loadRequested) {
        const bool saving = state_.controls.saveRequested;
        state_.controls.saveRequested = false;
        state_.controls.loadRequested = false;
        context.vulkan.waitIdle();
        try {
            if (saving) {
                saveWorldSnapshot(state_.controls.snapshotPath, driver_.snapshot());
                state_.controls.snapshotStatus = "Saved " + state_.controls.snapshotPath;
            } else {
                driver_.restoreSnapshot(loadWorldSnapshot(state_.controls.snapshotPath));
                state_.controls.snapshotStatus = "Loaded " + state_.controls.snapshotPath;
                finishPending_ = false;
            }
        } catch (const std::exception& error) {
            state_.controls.snapshotStatus = error.what();
        }
        return;
    }
    if (state_.controls.resetRequested) {
        context.vulkan.waitIdle();
        driver_.restart();
        state_.controls.resetRequested = false;
        finishPending_ = false;
    } else if (finishPending_) {
        context.vulkan.waitIdle();
        driver_.finishGeneration();
        finishPending_ = false;
    }
}

void SimulationModule::onRender(AppContext& context, const FrameInfo&) {
    if (state_.controls.paused || finishPending_) {
        return;
    }
    auto cpuScope = context.profiler.cpu().scope(metric_);
    auto gpuScope = context.profiler.gpu().scope(context.vulkan.commandBuffer(), metric_);
    if (driver_.recordSteps(context.vulkan.commandBuffer(), state_.controls.stepsPerFrame) == 0) {
        finishPending_ = true;
        return;
    }
    finishPending_ = driver_.generationComplete();
}

void SimulationModule::onDetach(AppContext&) { driver_.destroyResources(); }

} // namespace vkexp
