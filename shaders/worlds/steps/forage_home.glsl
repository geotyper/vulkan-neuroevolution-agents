// floats1 = {pickup reward, delivery reward, unused, unused}.
// Mirrors beforeStep in src/worlds/scenarios/ForageHomeScenario.cpp.
void forageHomeScenarioBeforeStep(inout Agent agent) {
    if (agent.internal.y >= 0.5 &&
        forageHomeScenarioRelocated(params.scenario, params.deltaTime)) {
        scenarioBankProgress(agent, true);
    }
}

// Mirrors afterStep in src/worlds/scenarios/ForageHomeScenario.cpp.
// Measured from the nearest beacon, as this scenario always has -- see
// scenarioDeliveryCycleAfterStep.
void forageHomeScenarioAfterStep(inout Agent agent, float distance) {
    scenarioDeliveryCycleAfterStep(agent, distance, false);
}
