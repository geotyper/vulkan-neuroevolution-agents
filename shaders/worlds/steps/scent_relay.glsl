// Mirrors afterStep in src/worlds/scenarios/ScentRelayScenario.cpp. There is no
// beforeStep: home never moves, so no banked progress can be invalidated.
// Measured from the nearest beacon, as this scenario always has -- see
// scenarioDeliveryCycleAfterStep.
void scentRelayScenarioAfterStep(inout Agent agent, float distance) {
    scenarioDeliveryCycleAfterStep(agent, distance, false);
}
