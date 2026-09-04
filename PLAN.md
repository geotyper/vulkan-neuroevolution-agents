# Neuroevolution Lab Plan

## Design rules

1. Domain algorithms remain usable without a window or Vulkan device.
2. GPU layouts have explicit size/offset assertions and a CPU reference.
3. Modules exchange published views and state, not ownership of each other's
   implementation details.
4. Add one evolutionary pressure at a time and keep deterministic replay tests.
5. Prefer measurable behavioral milestones over adding simulation features in
   parallel.

## Runtime flow

```text
ImGui controls
     |
SimulationState
     |
CPU generation boundary: fitness -> GA -> genomes/initial agents
     |
GPU per-step: spatial grid -> RGB/touch receptors -> brain -> collisions/physics
     |
storage barrier
     |
AgentRenderer -> off-screen image -> ImGui
```

Generation transitions are intentionally synchronous in the first version.
The GPU performs the expensive per-agent/per-step work; the CPU reads results
and evolves 512 small genomes only once per generation. This is inspectable and
easy to validate before asynchronous readback or GPU-side selection is added.

## Milestones

### 1. Phototaxis foundation — complete

- [x] fixed flattened neural topology and CPU evaluator;
- [x] std430 agent/parameter contracts;
- [x] CPU reference receptors, physics, and fitness;
- [x] GPU sensor/network/physics compute shader;
- [x] population with multiple trials per genome;
- [x] elitism, tournament selection, crossover, and mutation;
- [x] independent SSBO visualization and ImGui controls;
- [x] headless CPU/GPU one-step parity test;
- [x] unit tests and validation-layer launch smoke test.

Exit criterion: best and median fitness can be observed across generations,
and CPU/GPU parity fails loudly after a contract-breaking shader change.

### 2. Diagnostics and replay

- [ ] champion-only replay with fixed seeds;
- [x] best/median/mean fitness and arrival-history plots;
- [ ] generation timing;
- [ ] inspect one agent's receptor values, activations, and motor outputs;
- [ ] render photoreceptor rays;
- [ ] save/load versioned genome files;
- [ ] deterministic multi-step CPU/GPU regression cases.

### 3. Geometry and navigation

- [ ] world interface for circles, segments, and material properties;
- [ ] extend the spatial grid with wall segments and ray queries;
- [ ] wall distance/type receptor channels;
- [x] per-trial GPU spatial grid for agent queries;
- [x] circle-circle collision and tactile interaction tests;
- [ ] wall-ray and occlusion parity tests;
- [ ] procedural maze trials with train/evaluation seed separation.

### 4. Memory and task switching

- [ ] recurrent/internal state with explicit reset semantics;
- [ ] energy pickup and nest delivery;
- [ ] outbound/return behavior fitness;
- [ ] ablation mode comparing reactive and recurrent brains.

### 5. Emergent communication

- [x] spatially accelerated additive RGB perception of nearby agents;
- [x] signal energy cost;
- [ ] wall occlusion;
- [ ] family/colony fitness and related genome batches;
- [ ] signal-off ablation to prove communication affects fitness;
- [ ] multiple colonies and optional interception of foreign signals.

### 6. Scale and extensibility

- [ ] asynchronous double-buffered generation readback;
- [ ] optional GPU selection/mutation backend;
- [ ] pluggable topology descriptors and recurrent evaluator;
- [ ] scenario registry and data-driven experiment configuration;
- [ ] batch/headless evolution executable;
- [ ] shader hot reload and capture/replay tooling.

## Immediate next step

Add champion replay and receptor visualization before walls. Those tools make
later failures attributable to perception, control, fitness, or evolution
instead of only showing that population fitness stopped improving.
