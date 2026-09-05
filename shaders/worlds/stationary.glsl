vec2 stationaryScenarioPosition(Agent agent) {
    return agent.target.xy;
}

vec3 stationaryScenarioColor(uint trial) {
    return trialPaletteColor(trial);
}
