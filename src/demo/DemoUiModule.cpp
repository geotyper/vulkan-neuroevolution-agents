#include "vkexp/demo/DemoUiModule.hpp"

#include "vkexp/demo/DemoState.hpp"
#include "vkexp/profiling/Profiler.hpp"
#include "vkexp/ui/ImGuiModule.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace vkexp {

DemoUiModule::DemoUiModule(DemoState& state, ImGuiModule& imgui, Profiler& profiler)
    : state_(state), imgui_(imgui), metric_(profiler.registerMetric("Demo UI")) {}

void DemoUiModule::onAttach(AppContext&) { syncTextures(); }

void DemoUiModule::syncTextures() {
    if (viewportGeneration_ != state_.viewport.generation) {
        imgui_.removeTexture(viewportDescriptor_);
        viewportDescriptor_ = imgui_.addTexture(state_.viewport.sampler, state_.viewport.imageView,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        viewportGeneration_ = state_.viewport.generation;
    }
    if (blurGeneration_ != state_.blur.generation) {
        imgui_.removeTexture(blurDescriptor_);
        blurDescriptor_ = imgui_.addTexture(state_.blur.sampler, state_.blur.imageView,
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        blurGeneration_ = state_.blur.generation;
    }
}

void DemoUiModule::onUpdate(AppContext& context, const FrameInfo& frame) {
    auto cpuScope = context.profiler.cpu().scope(metric_);
    syncTextures();

    ImGui::SetNextWindowPos(ImVec2(20.0F, 20.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320.0F, 250.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls");
    ImGui::Text("Preset: %s", state_.preset.name.c_str());
    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(frame.frameNumber));
    ImGui::Text("%.2f ms (%.1f FPS)", frame.deltaSeconds * 1000.0F,
                frame.deltaSeconds > 0.0F ? 1.0F / frame.deltaSeconds : 0.0F);
    ImGui::Separator();
    ImGui::Checkbox("Graphics pipeline", &state_.preset.graphicsEnabled);
    ImGui::Checkbox("Compute pipeline", &state_.preset.computeEnabled);
    ImGui::ColorEdit3("Clear color", &state_.preset.clearColor.x);
    ImGui::SliderInt("Blur radius", &state_.blur.radius, 1, 12);
    ImGui::BeginDisabled(!state_.preset.computeEnabled);
    if (ImGui::Button("Start")) {
        state_.blur.requested = true;
        state_.blur.ready = false;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextUnformatted(state_.blur.ready ? "Result ready" : "Waiting");
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(360.0F, 20.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430.0F, 640.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Viewport");
    const ImVec2 available = ImGui::GetContentRegionAvail();
    if (available.x >= 64.0F && available.y >= 64.0F) {
        state_.viewport.requestedWidth = static_cast<std::uint32_t>(std::floor(available.x));
        state_.viewport.requestedHeight = static_cast<std::uint32_t>(std::floor(available.y));
        const ImTextureID texture =
            static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(viewportDescriptor_));
        ImGui::Image(texture, available);
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(810.0F, 20.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430.0F, 640.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Blur Output");
    const ImVec2 availableBlur = ImGui::GetContentRegionAvail();
    if (availableBlur.x > 1.0F && availableBlur.y > 1.0F && state_.blur.ready &&
        blurDescriptor_ != VK_NULL_HANDLE && state_.blur.extent.width > 0 &&
        state_.blur.extent.height > 0) {
        const float scale =
            std::min(availableBlur.x / static_cast<float>(state_.blur.extent.width),
                     availableBlur.y / static_cast<float>(state_.blur.extent.height));
        const ImVec2 imageSize{
            static_cast<float>(state_.blur.extent.width) * scale,
            static_cast<float>(state_.blur.extent.height) * scale,
        };
        const ImTextureID texture =
            static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(blurDescriptor_));
        ImGui::Image(texture, imageSize);
    } else {
        ImGui::TextDisabled("Press Start to run the compute blur.");
    }
    ImGui::End();

    profilerPanel_.draw(context.profiler);
}

void DemoUiModule::onDetach(AppContext&) {
    imgui_.removeTexture(blurDescriptor_);
    imgui_.removeTexture(viewportDescriptor_);
    blurDescriptor_ = VK_NULL_HANDLE;
    viewportDescriptor_ = VK_NULL_HANDLE;
}

} // namespace vkexp
