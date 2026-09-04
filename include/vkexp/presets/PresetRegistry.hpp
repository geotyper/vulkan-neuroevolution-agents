#pragma once

#include "vkexp/presets/Preset.hpp"

#include <span>
#include <string_view>
#include <vector>

namespace vkexp {

class PresetRegistry {
public:
    PresetRegistry();

    void loadWindowPreset(std::string_view path);
    [[nodiscard]] const Preset& require(std::string_view name) const;
    [[nodiscard]] std::span<const Preset> all() const { return presets_; }

private:
    std::vector<Preset> presets_;
};

} // namespace vkexp

