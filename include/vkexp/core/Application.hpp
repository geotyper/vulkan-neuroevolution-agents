#pragma once

#include "vkexp/core/Module.hpp"
#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/core/Window.hpp"
#include "vkexp/profiling/Profiler.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vkexp {

struct ApplicationConfig {
    int windowWidth{1280};
    int windowHeight{720};
    std::string title{"Vulkan compute boilerplate"};
    bool validationEnabled{true};
};

class Application {
public:
    explicit Application(ApplicationConfig config = {});

    Module& addModule(std::unique_ptr<Module> module);
    [[nodiscard]] Profiler& profiler() { return profiler_; }
    [[nodiscard]] const Profiler& profiler() const { return profiler_; }
    int run();

private:
    Window window_;
    VulkanContext vulkan_;
    Profiler profiler_;
    AppContext context_;
    std::vector<std::unique_ptr<Module>> modules_;
    bool running_{};
};

} // namespace vkexp
