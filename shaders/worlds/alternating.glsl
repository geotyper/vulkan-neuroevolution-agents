// Geometry from the shared kernel; see include/vkexp/worlds/ScenarioKernel.inl.
vec2 alternatingScenarioPosition(uint beaconIndex, uint phase, float worldRadius) {
    const vec2 offset = alternatingDiagonalOffset(beaconIndex, phase);
    return offset * worldRadius;
}

vec3 alternatingScenarioColor(uint beaconIndex, uint phase) {
    return trialPaletteColor(phase * 2u + beaconIndex);
}
