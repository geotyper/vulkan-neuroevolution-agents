// The chain world's scoring, and the only place it exists.
//
// Every other world in this directory mirrors a hook in its C++ scenario, and
// the pair is kept in step by the parity test. This one has no twin: the number
// it reads, agent.target.w, is the neighbour count taken by the grid sweep in
// agent_step.comp, and there is no grid sweep on the CPU side to take it. So
// ChainScenario.cpp leaves afterStep null rather than carrying a copy that would
// compute zero and read as agreement. See the note at the top of that file.
//
// Per second and not per step: the score and the time in band are both weighted
// by deltaTime, so halving the step rate leaves a trial worth the same rather
// than half as much.
void chainScenarioAfterStep(inout Agent agent) {
    const uint neighbours = uint(max(agent.target.w, 0.0) + 0.5);
    const uint rewardBand = params.scenario.integers[0];
    const uint crowdLimit = params.scenario.integers[1];
    const float crowdPenalty = params.scenario.floats0.x;

    // Mirrors chain::stepScore in ChainScenario.cpp, which is where the rule is
    // stated for the test to check. The two are read together rather than shared
    // through an .inl because this is four lines of arithmetic on values that
    // have already crossed the boundary as parameters.
    const float reward = float(min(neighbours, rewardBand));
    const float crowd = float(neighbours > crowdLimit ? neighbours - crowdLimit : 0u);
    agent.metrics.w += (reward - crowdPenalty * crowd) * params.deltaTime;

    if (neighbours >= 1u && neighbours <= crowdLimit) {
        agent.penalties.y += params.deltaTime;
    }
}
