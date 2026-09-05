// Step hooks mirror the C++ ScenarioDefinition::beforeStep / afterStep pair.
// They live apart from worlds/<scenario>.glsl because they read the global
// `params`, which only agent_step.comp declares; the geometry files stay usable
// from the vertex shader.

// Mirrors vkexp::worlds::bankObjectiveProgress.
void scenarioBankProgress(inout Agent agent, bool clampToPositive) {
    const float progress = agent.metrics.x - agent.metrics.y;
    agent.metrics.w += clampToPositive ? max(progress, 0.0) : progress;
    agent.metrics.x = nearestBeaconDistance(agent, agent.pose.xy);
    agent.metrics.y = agent.metrics.x;
}

// Mirrors vkexp::worlds::recordPhaseArrival.
void scenarioRecordPhaseArrival(inout Agent agent, float distance) {
    if (distance >= params.arrivalRadius) {
        return;
    }
    const uint completedMask = uint(max(agent.target.w, 0.0) + 0.5) | (1u << params.beaconPhase);
    agent.target.w = float(completedMask);
}

// Mirrors vkexp::worlds::rewardVisibleTracking.
void scenarioRewardVisibleTracking(inout Agent agent, float distance) {
    const float visibleCloseness = clamp(1.0 - distance / params.lightSensorRange, 0.0, 1.0);
    agent.metrics.w += visibleCloseness * params.deltaTime * params.fitness.trackingReward;
}
