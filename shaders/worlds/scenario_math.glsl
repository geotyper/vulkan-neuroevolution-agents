const float Tau = 6.28318530718;

vec3 trialPaletteColor(uint index) {
    const vec3 colors[4] = vec3[](vec3(0.20, 0.85, 1.00), vec3(1.00, 0.35, 0.75),
                                  vec3(0.55, 1.00, 0.35), vec3(1.00, 0.72, 0.20));
    return colors[index % 4];
}

uint scenarioHash(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float scenarioRandom01(uint value) {
    return float(scenarioHash(value) & 0x00ffffffu) / 16777215.0;
}
