#include "vkexp/ui/SimulationUiModule.hpp"

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/profiling/Profiler.hpp"
#include "vkexp/ui/ImGuiModule.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace vkexp {
namespace {

void plotHistory(const char* label, const std::vector<float>& values, const float minimum = FLT_MAX,
                 const float maximum = FLT_MAX) {
    if (values.empty()) {
        ImGui::TextDisabled("%s: waiting for the first completed generation", label);
        return;
    }
    ImGui::PushID(label);
    ImGui::PlotLines("##history", values.data(), static_cast<int>(values.size()), 0, nullptr,
                     minimum, maximum, ImVec2(-1.0F, 62.0F));
    ImGui::TextDisabled("%s", label);
    ImGui::PopID();
}

} // namespace

SimulationUiModule::SimulationUiModule(SimulationState& state, ImGuiModule& imgui,
                                       Profiler& profiler)
    : state_(state), imgui_(imgui), metric_(profiler.registerMetric("Simulation UI")) {}

void SimulationUiModule::onAttach(AppContext&) { syncTexture(); }

void SimulationUiModule::syncTexture() {
    if (viewportGeneration_ == state_.viewport.generation) {
        return;
    }
    imgui_.removeTexture(viewportDescriptor_);
    viewportDescriptor_ = imgui_.addTexture(state_.viewport.sampler, state_.viewport.imageView,
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    viewportGeneration_ = state_.viewport.generation;
}

void SimulationUiModule::onUpdate(AppContext& context, const FrameInfo& frame) {
    auto cpuScope = context.profiler.cpu().scope(metric_);
    syncTexture();

    ImGui::SetNextWindowPos(ImVec2(16.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330.0F, 470.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Simulation");
    ImGui::Text("Generation %llu", static_cast<unsigned long long>(state_.statistics.generation));
    ImGui::ProgressBar(static_cast<float>(state_.statistics.step) /
                           static_cast<float>(std::max(state_.controls.stepsPerGeneration, 1U)),
                       ImVec2(-1.0F, 0.0F));
    ImGui::Text("GPU agents: %u  Genomes: %u x %u trials", state_.agents.agentCount,
                state_.agents.genomeCount, state_.agents.trialsPerGenome);
    ImGui::Text("Frame %.2f ms", frame.deltaSeconds * 1000.0F);
    ImGui::Checkbox("Pause", &state_.controls.paused);
    int stepsPerFrame = static_cast<int>(state_.controls.stepsPerFrame);
    if (ImGui::SliderInt("Steps / frame", &stepsPerFrame, 1, 32)) {
        state_.controls.stepsPerFrame = static_cast<std::uint32_t>(stepsPerFrame);
    }
    int generationSteps = static_cast<int>(state_.controls.stepsPerGeneration);
    if (ImGui::SliderInt("Steps / generation", &generationSteps, 120, 15000, "%d",
                         ImGuiSliderFlags_Logarithmic)) {
        state_.controls.stepsPerGeneration = static_cast<std::uint32_t>(generationSteps);
        state_.controls.resetRequested = true;
    }
    // The step stays the control, because a replay is reproduced by step count.
    // Seconds are shown beside it so arena size, speed and trial length can be
    // read against each other in the units they are actually expressed in.
    ImGui::Text("Trial %.1f s at %.1f Hz (dt %.2f ms)",
                static_cast<double>(units::secondsForSteps(state_.controls.stepsPerGeneration,
                                                           state_.physics.deltaTime)),
                static_cast<double>(1.0F / state_.physics.deltaTime),
                static_cast<double>(state_.physics.deltaTime * 1000.0F));
    ImGui::SeparatorText("World");
    int requestedAgentsPerWorld = static_cast<int>(state_.worlds.requestedAgentsPerWorld);
    const int populationSize = static_cast<int>(std::max(state_.agents.genomeCount, 1U));
    const int minimumGroupSize = std::min(static_cast<int>(minimumAgentsPerWorld), populationSize);
    if (ImGui::SliderInt("Agents / world", &requestedAgentsPerWorld, minimumGroupSize,
                         populationSize)) {
        state_.worlds.requestedAgentsPerWorld = static_cast<std::uint32_t>(requestedAgentsPerWorld);
        state_.controls.resetRequested = true;
    }
    if (ImGui::Button("12 agents")) {
        state_.worlds.requestedAgentsPerWorld = std::min(12U, state_.agents.genomeCount);
        state_.controls.resetRequested = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("All agents")) {
        state_.worlds.requestedAgentsPerWorld = state_.agents.genomeCount;
        state_.controls.resetRequested = true;
    }
    ImGui::TextDisabled("%u groups x %u trials = %u worlds", state_.worlds.groupCount,
                        state_.agents.trialsPerGenome, state_.worlds.worldCount);
    int worldSize = static_cast<int>(state_.physics.worldSize);
    constexpr const char* worldSizes[] = {"Small (x1)", "Medium (x1.5)", "Large (x3)"};
    if (ImGui::Combo("World size", &worldSize, worldSizes, 3)) {
        state_.physics.worldSize = static_cast<WorldSize>(worldSize);
        state_.physics.worldRadius = worldRadiusForSize(state_.physics.worldSize);
        state_.physics.lightSensorRange = lightRangeForWorld(state_.physics);
        state_.controls.resetRequested = true;
    }
    int worldShape = static_cast<int>(state_.physics.worldShape);
    constexpr const char* worldShapes[] = {"Circle", "Square"};
    if (ImGui::Combo("World shape", &worldShape, worldShapes, 2)) {
        state_.physics.worldShape = static_cast<WorldShape>(worldShape);
        state_.controls.resetRequested = true;
    }
    ImGui::Text("World span: %.2f", state_.physics.worldRadius * 2.0F);
    // Every scenario control below is driven by the scenario definition, so a new
    // scenario appears here without editing this file.
    const std::span<const ScenarioDefinition* const> registry = scenarioRegistry();
    int beaconScenario = static_cast<int>(state_.physics.beaconScenario);
    std::vector<const char*> beaconScenarios(registry.size());
    for (std::size_t index = 0; index < registry.size(); ++index) {
        beaconScenarios[index] = registry[index]->name;
    }
    if (ImGui::Combo("Beacon scenario", &beaconScenario, beaconScenarios.data(),
                     static_cast<int>(beaconScenarios.size()))) {
        state_.physics.beaconScenario = static_cast<BeaconScenario>(beaconScenario);
        state_.controls.resetRequested = true;
    }
    const ScenarioDefinition& scenario = scenarioDefinition(state_.physics.beaconScenario);
    if (ImGui::SliderFloat("Arrival radius multiplier", &state_.physics.arrivalRadiusMultiplier,
                           0.1F, 5.0F, "x%.2f")) {
        state_.controls.resetRequested = true;
    }
    ImGui::TextDisabled("Arrival distance %.3f (x1 = beacon circle)",
                        beaconArrivalRadius(state_.physics));
    if (scenario.tunables.beaconRadiusRatio) {
        float beaconRadius = state_.physics.beaconRadiusRatio * state_.physics.worldRadius;
        if (ImGui::SliderFloat(scenario.radiusLabel, &beaconRadius,
                               state_.physics.worldRadius * 0.10F,
                               state_.physics.worldRadius * 0.90F, "%.2f")) {
            state_.physics.beaconRadiusRatio = beaconRadius / state_.physics.worldRadius;
            state_.controls.resetRequested = true;
        }
    }
    if (scenario.phaseForStep != nullptr) {
        const std::uint32_t phase = beaconPhaseForStep(scenario.id, state_.statistics.step,
                                                       state_.controls.stepsPerGeneration) +
                                    1U;
        ImGui::TextDisabled("Active phase: %u / %u", phase, scenario.objectivesPerAgent);
    }
    if (scenario.tunables.beaconAngularSpeed) {
        if (ImGui::SliderAngle("Rotation speed", &state_.physics.beaconAngularSpeed, -90.0F, 90.0F,
                               "%.1f deg/s")) {
            state_.controls.resetRequested = true;
        }
        if (std::abs(state_.physics.beaconAngularSpeed) > 0.0001F) {
            constexpr float tau = 6.28318530718F;
            ImGui::TextDisabled("Orbit period: %.1f s",
                                tau / std::abs(state_.physics.beaconAngularSpeed));
        }
    }
    if (scenario.tunables.beaconRandomMotion) {
        if (ImGui::SliderFloat("Wander speed", &state_.physics.beaconRandomSpeed, 0.0F, 0.75F,
                               "%.2f")) {
            state_.controls.resetRequested = true;
        }
        float teleportPercent = state_.physics.beaconTeleportProbability * 100.0F;
        if (ImGui::SliderFloat("Teleport chance", &teleportPercent, 0.0F, 100.0F, "%.0f%%")) {
            state_.physics.beaconTeleportProbability = teleportPercent * 0.01F;
            state_.controls.resetRequested = true;
        }
    }
    if (scenario.tunables.forageCargoDecay) {
        if (ImGui::SliderFloat("Cargo decay / second", &state_.physics.forageCargoDecayRate, 0.0F,
                               0.25F, "%.3f")) {
            state_.controls.resetRequested = true;
        }
        if (ImGui::SliderFloat("Pickup reward", &state_.physics.foragePickupReward, 0.0F, 2.0F,
                               "%.2f")) {
            state_.controls.resetRequested = true;
        }
        if (ImGui::SliderFloat("Delivery reward", &state_.physics.forageDeliveryReward, 0.0F, 12.0F,
                               "%.2f")) {
            state_.controls.resetRequested = true;
        }
        ImGui::TextDisabled("Home relocates every %.0f seconds", forageHomeRelocationSeconds);
    }
    if (scenario.description != nullptr) {
        ImGui::TextDisabled("%s", scenario.description);
    }
    // Shaping weights reach both the CPU reference and the shader as parameters,
    // so a fitness experiment is a slider rather than a rebuild.
    ImGui::SeparatorText("Fitness shaping");
    if (ImGui::SliderFloat("Objective bonus", &state_.physics.fitness.objectiveBonus, 0.0F, 10.0F,
                           "%.2f")) {
        state_.controls.resetRequested = true;
    }
    if (ImGui::SliderFloat("Motor cost weight", &state_.physics.fitness.motorCostWeight, 0.0F,
                           0.02F, "%.4f")) {
        state_.controls.resetRequested = true;
    }
    if (scenario.tunables.beaconAngularSpeed || scenario.tunables.beaconRandomMotion) {
        if (ImGui::SliderFloat("Tracking reward", &state_.physics.fitness.trackingReward, 0.0F,
                               1.0F, "%.3f")) {
            state_.controls.resetRequested = true;
        }
    }
    if (ImGui::SliderFloat("Signal cost factor", &state_.physics.fitness.signalCostFactor, 0.0F,
                           2.0F, "%.2f")) {
        state_.controls.resetRequested = true;
    }
    if (ImGui::SliderFloat("Energy drain", &state_.physics.fitness.energyDrain, 0.0F, 0.005F,
                           "%.4f")) {
        state_.controls.resetRequested = true;
    }

    ImGui::SeparatorText("Physics");
    ImGui::Text("Body %.1f cm across, arena %.2f m wide",
                static_cast<double>(units::metresToCentimetres(agentBodyRadius * 2.0F)),
                static_cast<double>(state_.physics.worldRadius * 2.0F));
    ImGui::SliderFloat("Thrust (m/s2)", &state_.physics.thrust, 0.2F, 4.0F);
    ImGui::SliderFloat("Turn (rad/s2)", &state_.physics.turnAcceleration, 0.5F, 10.0F);
    ImGui::SliderFloat("Linear drag (1/s)", &state_.physics.linearDrag, 0.1F, 5.0F);
    ImGui::SliderFloat("Angular drag (1/s)", &state_.physics.angularDrag, 0.1F, 6.0F);
    ImGui::SliderFloat("Maximum speed (m/s)", &state_.physics.maximumSpeed, 0.10F, 1.50F);
    ImGui::SliderFloat("Maximum turn speed (rad/s)", &state_.physics.maximumAngularSpeed, 0.25F,
                       8.0F);
    ImGui::SliderFloat("Collision restitution", &state_.physics.collisionRestitution, 0.0F, 1.0F);
    ImGui::SliderFloat("Contact stiffness (1/s)", &state_.physics.contactStiffness, 5.0F, 300.0F);
    ImGui::SetItemTooltip(
        "Overlap resolved per step: %.0f%%",
        static_cast<double>(100.0F * (1.0F - std::exp(-state_.physics.contactStiffness *
                                                      state_.physics.deltaTime))));
    ImGui::SliderFloat("Wall penalty / s", &state_.physics.wallCollisionPenalty, 0.0F, 6.0F,
                       "%.2f");
    ImGui::Checkbox("Agent collisions", &state_.physics.agentCollisionsEnabled);
    ImGui::SeparatorText("Trail field");
    if (ImGui::Checkbox("Leave trails", &state_.physics.trailEnabled)) {
        state_.controls.resetRequested = true;
    }
    ImGui::SliderFloat("Trail deposit / s", &state_.physics.trailDepositRate, 0.0F, 24.0F, "%.2f");
    ImGui::SliderFloat("Beacon deposit / s", &state_.physics.beaconTrailDepositRate, 0.0F, 64.0F,
                       "%.2f");
    ImGui::SliderFloat("Trail half-life (s)", &state_.physics.trailHalfLife, 0.25F, 30.0F, "%.2f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::TextDisabled(
        "%u x %u cells of %.0f cm per world", trailWidthForWorld(state_.physics.worldRadius),
        trailWidthForWorld(state_.physics.worldRadius),
        static_cast<double>(units::metresToCentimetres(trail::kernel::TrailCellSize)));
    ImGui::SeparatorText("Sensors");
    ImGui::SliderAngle("Sensor FOV", &state_.physics.sensorFieldOfView, 20.0F, 170.0F);
    ImGui::SliderFloat("Light range (m)", &state_.physics.lightSensorRange, 0.25F,
                       state_.physics.worldRadius * 2.0F);
    ImGui::SliderFloat("Light exposure", &state_.physics.lightExposure, 0.1F, 4.0F);
    ImGui::Checkbox("Perceive agent light", &state_.physics.agentLightEnabled);
    ImGui::SeparatorText("Brain contract");
    const neuro::BrainShape brain = scenarioDefinition(state_.physics.beaconScenario).brain;
    ImGui::Text("%zu inputs -> %zu tanh -> %zu outputs", brain.inputCount, brain.hiddenCount,
                brain.outputCount);
    ImGui::TextDisabled("%zu active weights / %zu genome capacity", brain.weightCount(),
                        neuro::Topology::weightCount);
    if (brain.outputCount > neuro::Topology::actuatorOutputCount) {
        ImGui::TextDisabled("light, touch, self, task state and recurrent memory");
        ImGui::TextDisabled("motors, RGB/light intensity and memory updates");
    } else {
        ImGui::TextDisabled("light, touch and self state");
        ImGui::TextDisabled("motors and RGB/light intensity");
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(16.0F, 500.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330.0F, 380.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Genetic Algorithm");
    if (ImGui::Button("Reset evolution")) {
        state_.controls.resetRequested = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("completed: %zu", state_.history.bestFitness.size());
    ImGui::SeparatorText("Last evaluated generation");
    ImGui::Text("Best fitness:   %.4f", state_.statistics.bestFitness);
    ImGui::Text("Median fitness: %.4f", state_.statistics.medianFitness);
    ImGui::Text("Mean fitness:   %.4f", state_.statistics.meanFitness);
    ImGui::Text("%s: %.1f%%", scenarioDefinition(state_.physics.beaconScenario).objectiveLabel,
                state_.statistics.arrivalRatio * 100.0F);
    ImGui::SeparatorText("Fitness history");
    plotHistory("Best", state_.history.bestFitness);
    plotHistory("Median", state_.history.medianFitness);
    plotHistory("Mean", state_.history.meanFitness);
    plotHistory("Objective completion", state_.history.arrivalRatio, 0.0F, 1.0F);
    ImGui::SeparatorText("Evolution parameters");
    ImGui::Text("Population: %zu", state_.evolution.populationSize);
    ImGui::Text("Elites: %zu   Tournament: %zu", state_.evolution.eliteCount,
                state_.evolution.tournamentSize);
    ImGui::Text("Crossover: %.1f%%", state_.evolution.crossoverProbability * 100.0F);
    ImGui::Text("Mutation: %.1f%%  strength %.3f", state_.evolution.mutationProbability * 100.0F,
                state_.evolution.mutationStrength);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(360.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1040.0F, 820.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Agent World");
    if (state_.worlds.worldCount > 0) {
        int visibleWorld = static_cast<int>(state_.worlds.selectedWorld + 1);
        if (ImGui::SliderInt("Visible world", &visibleWorld, 1,
                             static_cast<int>(state_.worlds.worldCount))) {
            state_.worlds.selectedWorld = static_cast<std::uint32_t>(visibleWorld - 1);
        }
        const std::uint32_t visibleGroup =
            state_.worlds.selectedWorld / state_.agents.trialsPerGenome;
        const std::uint32_t visibleTrial =
            state_.worlds.selectedWorld % state_.agents.trialsPerGenome;
        const std::uint32_t visibleAgentCount =
            agentsInLogicalWorld(state_.agents.genomeCount, state_.worlds.agentsPerWorld,
                                 state_.agents.trialsPerGenome, state_.worlds.selectedWorld);
        ImGui::TextDisabled("Group %u / %u, trial %u / %u, %u agents", visibleGroup + 1,
                            state_.worlds.groupCount, visibleTrial + 1,
                            state_.agents.trialsPerGenome, visibleAgentCount);
    }
    const ImVec2 available = ImGui::GetContentRegionAvail();
    if (available.x >= 64.0F && available.y >= 64.0F) {
        state_.viewport.requestedWidth = static_cast<std::uint32_t>(std::floor(available.x));
        state_.viewport.requestedHeight = static_cast<std::uint32_t>(std::floor(available.y));
        const ImTextureID texture =
            static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(viewportDescriptor_));
        ImGui::Image(texture, available);
    }
    ImGui::End();
    profilerPanel_.draw(context.profiler);
}

void SimulationUiModule::onDetach(AppContext&) {
    imgui_.removeTexture(viewportDescriptor_);
    viewportDescriptor_ = VK_NULL_HANDLE;
}

} // namespace vkexp
