// Mirrors afterStep in src/worlds/scenarios/ScentRelayScenario.cpp. There is no
// beforeStep: home never moves, so no banked progress can be invalidated.
void scentRelayScenarioAfterStep(inout Agent agent, float distance) {
    scenarioDeliveryCycleAfterStep(agent, distance);
}
