#include "vkexp/compute/ComputeModule.hpp"
#include "vkexp/core/Application.hpp"
#include "vkexp/demo/DemoState.hpp"
#include "vkexp/demo/DemoUiModule.hpp"
#include "vkexp/graphics/GraphicsModule.hpp"
#include "vkexp/presets/PresetRegistry.hpp"
#include "vkexp/ui/ImGuiModule.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

namespace {

void printHelp(const char* executable) {
    std::cout << "Usage: " << executable << " [--preset NAME] [--no-validation]\n"
              << "       " << executable << " --list-presets\n";
}

} // namespace

int main(const int argc, char** argv) {
    try {
        vkexp::PresetRegistry presets;
        std::string presetName = "mixed";
#ifdef VKEXP_ENABLE_VALIDATION
        bool validationEnabled = true;
#else
        bool validationEnabled = false;
#endif

        for (int i = 1; i < argc; ++i) {
            const std::string_view argument = argv[i];
            if (argument == "--preset") {
                if (++i >= argc) {
                    throw std::runtime_error("--preset requires a name");
                }
                presetName = argv[i];
            } else if (argument == "--no-validation") {
                validationEnabled = false;
            } else if (argument == "--list-presets") {
                for (const auto& preset : presets.all()) {
                    std::cout << preset.name << "\t" << preset.description << '\n';
                }
                return 0;
            } else if (argument == "--help" || argument == "-h") {
                printHelp(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + std::string{argument});
            }
        }

        presets.loadWindowPreset(VKEXP_WINDOW_PRESET);
        vkexp::DemoState state{presets.require(presetName)};
        vkexp::Application app{vkexp::ApplicationConfig{
            state.preset.windowWidth,
            state.preset.windowHeight,
            "Vulkan compute boilerplate",
            validationEnabled,
        }};

        auto imgui = std::make_unique<vkexp::ImGuiModule>(app.profiler());
        auto& imguiBackend = *imgui;
        app.addModule(std::make_unique<vkexp::GraphicsModule>(state, app.profiler()));
        app.addModule(std::make_unique<vkexp::ComputeModule>(state, app.profiler()));
        app.addModule(std::move(imgui));
        app.addModule(std::make_unique<vkexp::DemoUiModule>(state, imguiBackend, app.profiler()));
        return app.run();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
