// Mirrors beforeStep in src/worlds/scenarios/AlternatingScenario.cpp.
void alternatingScenarioBeforeStep(inout Agent agent) {
    if (params.beaconPhaseChanged != 0) {
        scenarioBankProgress(agent, false);
    }
}
