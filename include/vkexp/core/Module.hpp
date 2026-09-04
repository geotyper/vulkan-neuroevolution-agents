#pragma once

#include <cstdint>

namespace vkexp {

class VulkanContext;
class Window;
class Profiler;

struct FrameInfo {
    float deltaSeconds{};
    float elapsedSeconds{};
    std::uint64_t frameNumber{};
};

struct AppContext {
    Window& window;
    VulkanContext& vulkan;
    Profiler& profiler;
};

class Module {
public:
    virtual ~Module() = default;

    virtual void onAttach(AppContext&) {}
    virtual void onFrameBegin(AppContext&, const FrameInfo&) {}
    virtual void onUpdate(AppContext&, const FrameInfo&) {}
    virtual void onRender(AppContext&, const FrameInfo&) {}
    virtual void onFrameEnd(AppContext&, const FrameInfo&) {}
    virtual void onDetach(AppContext&) {}
};

} // namespace vkexp
