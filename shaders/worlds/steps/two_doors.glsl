// Mirrors afterStep in src/worlds/scenarios/TwoDoorsScenario.cpp. The delivery
// cycle is the foraging one; what differs is only the wall between the two ends.
// Measured from the target that has just become current, so crossing the
// wall toward it is rewarded as progress.
void twoDoorsScenarioAfterStep(inout Agent agent, float distance) {
    scenarioDeliveryCycleAfterStep(agent, distance, true);
}
