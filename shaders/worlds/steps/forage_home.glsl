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
    scenarioDeliveryCycleAfterStep(agent, distance);
}
