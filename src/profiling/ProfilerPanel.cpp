#include "vkexp/profiling/ProfilerPanel.hpp"

#include "vkexp/profiling/Profiler.hpp"

#include <imgui.h>

#include <array>
#include <cfloat>
#include <cstdio>
#include <span>

namespace vkexp {
namespace {

template <typename Backend>
void drawStatisticsTable(const char* label, const Backend& backend,
                         const std::span<const std::string> metricNames) {
    if (!ImGui::BeginTable(label, 6,
                           ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                               ImGuiTableFlags_SizingStretchProp)) {
        return;
    }

    ImGui::TableSetupColumn("Metric");
    ImGui::TableSetupColumn("Current ms");
    ImGui::TableSetupColumn("Average ms");
    ImGui::TableSetupColumn("Minimum ms");
    ImGui::TableSetupColumn("Maximum ms");
    ImGui::TableSetupColumn("P95 ms");
    ImGui::TableHeadersRow();

    for (std::size_t index = 0; index < metricNames.size(); ++index) {
        const auto metric = static_cast<ProfileMetricId>(index);
        const TimingStatistics statistics = backend.series(metric).statistics();
        if (statistics.sampleCount == 0) {
            continue;
        }
        ImGui::TableSetColumnIndex(0);
        ImGui::TableNextRow();
        ImGui::TextUnformatted(metricNames[index].c_str());
        const std::array values{
            statistics.currentMs, statistics.averageMs,      statistics.minimumMs,
            statistics.maximumMs, statistics.percentile95Ms,
        };
        for (int column = 1; column < 6; ++column) {
            ImGui::TableSetColumnIndex(column);
            ImGui::Text("%.3f", values[static_cast<std::size_t>(column - 1)]);
        }
    }
    ImGui::EndTable();
}

void drawHistoryPlot(const char* label, const TimingSeries& series) {
    const std::span<const float> values = series.values();
    if (values.empty()) {
        ImGui::TextDisabled("%s: no samples", label);
        return;
    }

    const TimingStatistics statistics = series.statistics();
    std::array<char, 64> overlay{};
    std::snprintf(overlay.data(), overlay.size(), "current %.3f ms | p95 %.3f ms",
                  statistics.currentMs, statistics.percentile95Ms);
    ImGui::PlotLines(label, values.data(), static_cast<int>(values.size()), 0, overlay.data(),
                     FLT_MAX, FLT_MAX, ImVec2(0.0F, 80.0F));
}

} // namespace

void ProfilerPanel::draw(const Profiler& profiler) {
    ImGui::SetNextWindowPos(ImVec2(20.0F, 290.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(760.0F, 400.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Profiler");

    if (ImGui::CollapsingHeader("CPU timings", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawStatisticsTable("CPU statistics", profiler.cpu(), profiler.metricNames());
        const FrameDiagnostics& diagnostics = profiler.frameDiagnostics();
        const TimingStatistics wall = profiler.cpu().series(profiler.cpuFrameMetric()).statistics();
        const TimingStatistics work = profiler.cpu().series(profiler.cpuWorkMetric()).statistics();
        const double averageCpuLoad =
            wall.averageMs > 0.0F ? static_cast<double>(work.averageMs) / wall.averageMs * 100.0
                                  : 0.0;
        ImGui::Text("Process CPU load: %.1f%% current | %.1f%% rolling average",
                    diagnostics.currentCpuLoadPercent, averageCpuLoad);
        if (diagnostics.refreshRateHz > 0.0) {
            ImGui::Text("Display: %.0f Hz | estimated missed VSync: %u last | %llu total",
                        diagnostics.refreshRateHz, diagnostics.estimatedMissedVsyncs,
                        static_cast<unsigned long long>(diagnostics.totalEstimatedMissedVsyncs));
        } else {
            ImGui::TextDisabled(
                "Display refresh rate unavailable; missed VSync estimate disabled.");
        }
    }

    if (ImGui::CollapsingHeader("GPU timings", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (profiler.gpu().supported()) {
            drawStatisticsTable("GPU statistics", profiler.gpu(), profiler.metricNames());
        } else {
            ImGui::TextDisabled("Timestamp queries are unavailable on this device.");
        }
    }

    ImGui::SeparatorText("120-sample history");
    const auto names = profiler.metricNames();
    if (selectedMetric_ >= static_cast<int>(names.size())) {
        selectedMetric_ = 0;
    }
    if (ImGui::BeginCombo("Metric", names[static_cast<std::size_t>(selectedMetric_)].c_str())) {
        for (std::size_t index = 0; index < names.size(); ++index) {
            const bool selected = selectedMetric_ == static_cast<int>(index);
            if (ImGui::Selectable(names[index].c_str(), selected)) {
                selectedMetric_ = static_cast<int>(index);
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    const auto metric = static_cast<ProfileMetricId>(selectedMetric_);
    if (!profiler.cpu().series(metric).values().empty()) {
        drawHistoryPlot("CPU history", profiler.cpu().series(metric));
    }
    if (profiler.gpu().supported()) {
        if (!profiler.gpu().series(metric).values().empty()) {
            drawHistoryPlot("GPU history", profiler.gpu().series(metric));
        }
    }

    ImGui::End();
}

} // namespace vkexp
