#include "vkexp/ui/SimulationUiModule.hpp"

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/profiling/Profiler.hpp"
#include "vkexp/ui/ImGuiModule.hpp"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

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
    ImGui::Text("Agents: %u  Genomes: %u x %u trials", state_.agents.agentCount,
                state_.agents.genomeCount, state_.agents.trialsPerGenome);
    ImGui::Text("Frame %.2f ms", frame.deltaSeconds * 1000.0F);
    ImGui::Checkbox("Pause", &state_.controls.paused);
    int stepsPerFrame = static_cast<int>(state_.controls.stepsPerFrame);
    if (ImGui::SliderInt("Steps / frame", &stepsPerFrame, 1, 32)) {
        state_.controls.stepsPerFrame = static_cast<std::uint32_t>(stepsPerFrame);
    }
    int generationSteps = static_cast<int>(state_.controls.stepsPerGeneration);
    if (ImGui::SliderInt("Steps / generation", &generationSteps, 120, 3600)) {
        state_.controls.stepsPerGeneration = static_cast<std::uint32_t>(generationSteps);
    }
    ImGui::SeparatorText("Physics");
    int worldShape = static_cast<int>(state_.physics.worldShape);
    constexpr const char* worldShapes[] = {"Circle", "Square"};
    if (ImGui::Combo("World shape", &worldShape, worldShapes, 2)) {
        state_.physics.worldShape = static_cast<WorldShape>(worldShape);
        state_.controls.resetRequested = true;
    }
    ImGui::Text("World span: %.2f", state_.physics.worldRadius * 2.0F);
    ImGui::SliderFloat("Thrust", &state_.physics.thrust, 0.2F, 4.0F);
    ImGui::SliderFloat("Turn", &state_.physics.turnAcceleration, 0.5F, 10.0F);
    ImGui::SliderFloat("Linear drag", &state_.physics.linearDrag, 0.1F, 5.0F);
    ImGui::SliderFloat("Angular drag", &state_.physics.angularDrag, 0.1F, 6.0F);
    ImGui::SliderFloat("Maximum speed", &state_.physics.maximumSpeed, 0.10F, 1.50F);
    ImGui::SliderFloat("Maximum turn speed", &state_.physics.maximumAngularSpeed, 0.25F, 8.0F);
    ImGui::SliderFloat("Collision restitution", &state_.physics.collisionRestitution, 0.0F, 1.0F);
    ImGui::SliderFloat("Collision stiffness", &state_.physics.collisionStiffness, 0.1F, 1.0F);
    ImGui::Checkbox("Agent collisions", &state_.physics.agentCollisionsEnabled);
    ImGui::SeparatorText("Sensors");
    ImGui::SliderAngle("Sensor FOV", &state_.physics.sensorFieldOfView, 20.0F, 170.0F);
    ImGui::SliderFloat("Light range", &state_.physics.lightSensorRange, 0.25F,
                       state_.physics.worldRadius * 2.0F);
    ImGui::SliderFloat("Light exposure", &state_.physics.lightExposure, 0.1F, 4.0F);
    ImGui::Checkbox("Perceive agent light", &state_.physics.agentLightEnabled);
    ImGui::SeparatorText("Brain contract");
    ImGui::Text("%zu inputs -> %zu tanh -> %zu outputs", neuro::Topology::inputCount,
                neuro::Topology::hiddenCount, neuro::Topology::outputCount);
    ImGui::TextDisabled("7 x RGB+luminance, 8 x wall+agent touch, 4 self");
    ImGui::TextDisabled("outputs: left/right motor, RGB, light intensity");
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
    ImGui::Text("Reached beacon: %.1f%%", state_.statistics.arrivalRatio * 100.0F);
    ImGui::SeparatorText("Fitness history");
    plotHistory("Best", state_.history.bestFitness);
    plotHistory("Median", state_.history.medianFitness);
    plotHistory("Mean", state_.history.meanFitness);
    plotHistory("Arrival ratio", state_.history.arrivalRatio, 0.0F, 1.0F);
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
