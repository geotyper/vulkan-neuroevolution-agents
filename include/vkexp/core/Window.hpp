#pragma once

#include <string_view>

struct GLFWwindow;

namespace vkexp {

class Window {
public:
    Window(int width, int height, std::string_view title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] bool shouldClose() const;
    [[nodiscard]] int refreshRateHz() const;
    void pollEvents() const;
    void waitForVisibleFramebuffer() const;
    [[nodiscard]] GLFWwindow* handle() const { return window_; }

private:
    GLFWwindow* window_{};
};

} // namespace vkexp
