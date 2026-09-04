#include "vkexp/core/Application.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>

namespace vkexp {

Application::Application(ApplicationConfig config)
    : window_(config.windowWidth, config.windowHeight, config.title),
      vulkan_(window_, config.validationEnabled), profiler_(vulkan_),
      context_{window_, vulkan_, profiler_} {}

Module& Application::addModule(std::unique_ptr<Module> module) {
    if (running_) {
        throw std::logic_error("Modules cannot be added while the application is running");
    }
    if (!module) {
        throw std::invalid_argument("Cannot add a null module");
    }
    Module& reference = *module;
    modules_.push_back(std::move(module));
    return reference;
}

int Application::run() {
    if (running_) {
        throw std::logic_error("Application is already running");
    }
    running_ = true;
    std::size_t attached = 0;
    try {
        for (auto& module : modules_) {
            module->onAttach(context_);
            ++attached;
        }

        using Clock = std::chrono::steady_clock;
        const auto started = Clock::now();
        auto previous = started;
        std::uint64_t frameNumber = 0;
        const double refreshRateHz = static_cast<double>(window_.refreshRateHz());
        while (!window_.shouldClose()) {
            profiler_.beginCpuFrame();
            FrameSynchronizationTimings synchronizationTimings{};
            window_.pollEvents();
            const auto now = Clock::now();
            FrameInfo frame{
                std::chrono::duration<float>(now - previous).count(),
                std::chrono::duration<float>(now - started).count(),
                frameNumber++,
            };
            previous = now;
            for (auto& module : modules_) {
                module->onFrameBegin(context_, frame);
            }
            for (auto& module : modules_) {
                module->onUpdate(context_, frame);
            }
            if (!vulkan_.beginFrame(&synchronizationTimings)) {
                for (auto& module : modules_) {
                    module->onFrameEnd(context_, frame);
                }
                profiler_.recordFrameSynchronization(
                    synchronizationTimings.fenceWaitMs, synchronizationTimings.acquireWaitMs,
                    synchronizationTimings.queueSubmitMs, synchronizationTimings.presentMs);
                profiler_.endCpuFrame(refreshRateHz);
                continue;
            }
            profiler_.beginGpuFrame(vulkan_.commandBuffer());
            for (auto& module : modules_) {
                module->onRender(context_, frame);
            }
            profiler_.endGpuFrame(vulkan_.commandBuffer());
            vulkan_.endFrame(&synchronizationTimings);
            for (auto& module : modules_) {
                module->onFrameEnd(context_, frame);
            }
            profiler_.recordFrameSynchronization(
                synchronizationTimings.fenceWaitMs, synchronizationTimings.acquireWaitMs,
                synchronizationTimings.queueSubmitMs, synchronizationTimings.presentMs);
            profiler_.endCpuFrame(refreshRateHz);
        }
        vulkan_.waitIdle();
    } catch (...) {
        vulkan_.waitIdle();
        for (std::size_t i = attached; i > 0; --i) {
            modules_[i - 1]->onDetach(context_);
        }
        running_ = false;
        throw;
    }

    for (auto iterator = modules_.rbegin(); iterator != modules_.rend(); ++iterator) {
        (*iterator)->onDetach(context_);
    }
    running_ = false;
    return 0;
}

} // namespace vkexp
