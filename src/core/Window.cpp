#include "vkexp/core/Window.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace vkexp {

Window::Window(const int width, const int height, const std::string_view title) {
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("GLFW initialization failed");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    const std::string ownedTitle{title};
    window_ = glfwCreateWindow(width, height, ownedTitle.c_str(), nullptr, nullptr);
    if (window_ == nullptr) {
        glfwTerminate();
        throw std::runtime_error("GLFW window creation failed");
    }
}

Window::~Window() {
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
    }
    glfwTerminate();
}

bool Window::shouldClose() const { return glfwWindowShouldClose(window_) == GLFW_TRUE; }
int Window::refreshRateHz() const {
    int windowX = 0;
    int windowY = 0;
    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowPos(window_, &windowX, &windowY);
    glfwGetWindowSize(window_, &windowWidth, &windowHeight);

    int monitorCount = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&monitorCount);
    GLFWmonitor* bestMonitor = nullptr;
    int bestOverlap = 0;
    for (int index = 0; index < monitorCount; ++index) {
        int monitorX = 0;
        int monitorY = 0;
        glfwGetMonitorPos(monitors[index], &monitorX, &monitorY);
        const GLFWvidmode* mode = glfwGetVideoMode(monitors[index]);
        if (mode == nullptr) {
            continue;
        }
        const int overlapWidth =
            std::max(0, std::min(windowX + windowWidth, monitorX + mode->width) -
                            std::max(windowX, monitorX));
        const int overlapHeight =
            std::max(0, std::min(windowY + windowHeight, monitorY + mode->height) -
                            std::max(windowY, monitorY));
        const int overlap = overlapWidth * overlapHeight;
        if (overlap > bestOverlap) {
            bestOverlap = overlap;
            bestMonitor = monitors[index];
        }
    }

    if (bestMonitor == nullptr) {
        bestMonitor = glfwGetPrimaryMonitor();
    }
    if (bestMonitor == nullptr) {
        return 0;
    }
    const GLFWvidmode* mode = glfwGetVideoMode(bestMonitor);
    return mode != nullptr ? mode->refreshRate : 0;
}

void Window::pollEvents() const { glfwPollEvents(); }

void Window::waitForVisibleFramebuffer() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window_, &width, &height);
    }
}

} // namespace vkexp
