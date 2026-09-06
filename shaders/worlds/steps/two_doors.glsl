// Mirrors afterStep in src/worlds/scenarios/TwoDoorsScenario.cpp. The delivery
// cycle is the foraging one; what differs is only the wall between the two ends.
void twoDoorsScenarioAfterStep(inout Agent agent, float distance) {
    scenarioDeliveryCycleAfterStep(agent, distance);
}
