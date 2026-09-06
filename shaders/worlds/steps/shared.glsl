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

// Mirrors vkexp::worlds::deliveryCycleAfterStep. Every scenario running the
// cycle packs the cargo decay rate into floats0.w and the pickup and delivery
// rewards into floats1.x and floats1.y, so this reads fixed slots rather than
// being told where they are.
void scenarioDeliveryCycleAfterStep(inout Agent agent, float distance,
                                    bool measureNextLegToTarget) {
    if (agent.internal.y >= 0.5) {
        agent.internal.x =
            max(0.0, agent.internal.x - params.scenario.floats0.w * params.deltaTime);
    }
    if (distance >= params.arrivalRadius) {
        return;
    }
    agent.metrics.w += max(agent.metrics.x - agent.metrics.y, 0.0);
    if (agent.internal.y >= 0.5) {
        agent.metrics.w += agent.internal.x * params.scenario.floats1.y;
        agent.target.w = floor(max(agent.target.w, 0.0) + 0.5) + 1.0;
        agent.internal.x = 0.0;
        agent.internal.y = 0.0;
    } else {
        agent.metrics.w += params.scenario.floats1.x;
        agent.internal.x = 1.0;
        agent.internal.y = 1.0;
    }
    agent.metrics.x =
        measureNextLegToTarget
            ? scenarioTargetDistance(params.beaconScenario, agent, agent.pose.xy,
                                     params.beaconPhase, params.worldRadius, params.beaconCount,
                                     params.scenario)
            : nearestBeaconDistance(agent, agent.pose.xy);
    agent.metrics.y = agent.metrics.x;
}

// Mirrors vkexp::worlds::rewardVisibleTracking.
void scenarioRewardVisibleTracking(inout Agent agent, float distance) {
    const float visibleCloseness = clamp(1.0 - distance / params.lightSensorRange, 0.0, 1.0);
    agent.metrics.w += visibleCloseness * params.deltaTime * params.fitness.trackingReward;
}
