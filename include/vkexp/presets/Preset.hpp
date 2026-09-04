#pragma once

#include <glm/vec4.hpp>

#include <string>

namespace vkexp {

struct Preset {
    std::string name;
    std::string description;
    bool graphicsEnabled{true};
    bool computeEnabled{true};
    glm::vec4 clearColor{0.025F, 0.035F, 0.055F, 1.0F};
    int windowWidth{1280};
    int windowHeight{720};
};

} // namespace vkexp

