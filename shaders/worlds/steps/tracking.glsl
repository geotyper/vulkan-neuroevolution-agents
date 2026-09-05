// Mirrors afterStep in RotatingScenario.cpp and RandomMovementScenario.cpp:
// a beacon that never stops needs continuous shaping, then the usual arrival.
void trackingScenarioAfterStep(inout Agent agent, float distance) {
    scenarioRewardVisibleTracking(agent, distance);
    scenarioRecordPhaseArrival(agent, distance);
}
