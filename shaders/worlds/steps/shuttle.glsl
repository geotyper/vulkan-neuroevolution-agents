// Mirrors afterStep in src/worlds/scenarios/ShuttleScenario.cpp. Measured from
// the target that has just become current, so setting off around the wall
// toward it is rewarded as progress.
void shuttleScenarioAfterStep(inout Agent agent, float distance) {
    scenarioDeliveryCycleAfterStep(agent, distance, true);
}
