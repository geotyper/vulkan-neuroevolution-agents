vec2 alternatingScenarioPosition(uint beaconIndex, uint phase, float worldRadius) {
    const float offset = worldRadius * 0.62;
    if (phase == 0u) {
        return beaconIndex == 0u ? vec2(-offset, -offset) : vec2(offset, offset);
    }
    return beaconIndex == 0u ? vec2(-offset, offset) : vec2(offset, -offset);
}

vec3 alternatingScenarioColor(uint beaconIndex, uint phase) {
    return trialPaletteColor(phase * 2u + beaconIndex);
}
