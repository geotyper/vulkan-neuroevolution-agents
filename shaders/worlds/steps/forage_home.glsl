// floats1 = {pickup reward, delivery reward, unused, unused}.
// Mirrors beforeStep in src/worlds/scenarios/ForageHomeScenario.cpp.
void forageHomeScenarioBeforeStep(inout Agent agent) {
    if (agent.internal.y >= 0.5 &&
        forageHomeScenarioRelocated(params.scenario, params.deltaTime)) {
        scenarioBankProgress(agent, true);
    }
}

// Mirrors afterStep in src/worlds/scenarios/ForageHomeScenario.cpp.
void forageHomeScenarioAfterStep(inout Agent agent, float distance) {
    if (agent.internal.y >= 0.5) {
        agent.internal.x =
            max(0.0, agent.internal.x -
                         forageHomeScenarioCargoDecayRate(params.scenario) * params.deltaTime);
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
    agent.metrics.x = nearestBeaconDistance(agent, agent.pose.xy);
    agent.metrics.y = agent.metrics.x;
}
