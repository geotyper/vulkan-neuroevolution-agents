// Mirrors afterStep in src/worlds/scenarios/TwoDoorsScenario.cpp. The delivery
// cycle is the foraging one; what differs is only the wall between the two ends.
void twoDoorsScenarioAfterStep(inout Agent agent, float distance) {
    if (agent.internal.y >= 0.5) {
        agent.internal.x =
            max(0.0, agent.internal.x -
                         twoDoorsScenarioCargoDecayRate(params.scenario) * params.deltaTime);
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
    agent.metrics.x = scenarioTargetDistance(params.beaconScenario, agent, agent.pose.xy,
                                             params.beaconPhase, params.worldRadius,
                                             params.beaconCount, params.scenario);
    agent.metrics.y = agent.metrics.x;
}
