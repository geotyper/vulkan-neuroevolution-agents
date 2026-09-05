# Neuroevolution Lab Plan

## Design rules

1. Domain algorithms remain usable without a window or Vulkan device.
2. GPU layouts have explicit size/offset assertions and a CPU reference.
3. Modules exchange published views and state, not ownership of each other's
   implementation details.
3a. Simulation logic lives in drivers, not in modules: anything needed for an
   experiment must be reachable without a window.
3b. A scenario is one contract, not a set of switches: adding one means adding a
   source file, a shader pair and a registry line.
3c. Anything the CPU and the GPU must agree on is either shared source or a
   runtime parameter -- never a constant written twice.
3d. The CPU path exists to inspect one agent and to build the network from the
   same declaration, not to mirror the GPU. Parallel behaviour is verified by
   tests that need many agents, not by a second implementation.
3e. Steps are the unit of reproducibility; seconds and metres are the units of
   physics. Anything the step accumulates over time is scaled by `deltaTime`,
   and any fraction removed per step is written as `1 - exp(-rate * dt)`.
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
- [x] save/load versioned genome files;
- [x] deterministic multi-step CPU/GPU regression cases;
- [x] accumulated CPU/GPU drift budget that catches sub-tolerance bias.

### 3. Geometry and navigation

- [ ] world interface for circles, segments, and material properties;
- [ ] extend the spatial grid with wall segments and ray queries;
- [ ] wall distance/type receptor channels;
- [x] per-trial GPU spatial grid for agent queries;
- [x] circle-circle collision and tactile interaction tests;
- [x] configurable fitness penalty for world-boundary contacts;
- [ ] wall-ray and occlusion parity tests;
- [ ] procedural maze trials with train/evaluation seed separation.

### 4. Memory and task switching

- [x] stationary and mid-generation alternating beacon scenarios;
- [x] rotating beacon scenario with adjustable angular speed;
- [x] deterministic random beacon paths with configurable teleport chance;
- [x] discrete small, medium, and large training arenas;
- [x] recurrent/internal state with explicit reset semantics;
- [x] energy pickup and nest delivery;
- [x] outbound/return behavior fitness;
- [x] runtime ablation switches reachable from the batch runner;
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
- [x] scenario-owned active dense topology descriptors and recurrent outputs;
- [x] one network preset driving the CPU evaluator, the shader and the tests;
- [ ] pluggable multi-layer/recurrent evaluator implementations;
- [x] scenario registry and data-driven experiment configuration;
- [x] batch/headless evolution executable;
- [ ] shader hot reload and capture/replay tooling.

## Runners

`vulkan_neuroevolution_agents` drives `SimulationDriver` from the frame loop.
`vkneuro_headless` drives the same driver from an `ImmediateContext`, so batch
sweeps and ablations produce results comparable with what the window shows:

```sh
vkneuro_headless --scenario rotating --generations 200 --csv runs/rotating.csv \
                 --save-champion runs/rotating-champion.vkng
vkneuro_headless --scenario forage --generations 200 --no-agent-light --quiet

# Fitness shaping is a parameter, so a sweep needs no rebuild.
for reward in 0.0 0.25 0.75; do
  vkneuro_headless --scenario rotating --generations 50 --seed 5 \
                   --tracking-reward "$reward" --csv "runs/tracking-$reward.csv"
done
```

## Changing the sensor suite or the brain

`include/vkexp/neuro/BrainKernel.inl` is the network preset. It declares how many
receptors, tactile sectors, self, task and recurrent inputs there are, how wide
the hidden layer is, and what the outputs mean; every offset, the input capacity,
the genome size and the packed GPU layout are derived from those numbers.

Both languages compile it, so raising `BrainSelfInputCount` from 4 to 5 moves the
CPU evaluator, the sensor sampler, the compute shader and the tests together, and
nothing else needs editing. The block offsets are asserted to tile the input
vector without gaps, so a sensor block that no longer fits fails loudly.

## Units

`include/vkexp/simulation/Units.hpp` declares the scale: one world unit is one
metre, the fixed rate is 60 Hz. The default arena is 3.68 m across, the body is
4.4 cm, top speed is 0.55 m/s and a 900-step trial is 15 seconds -- roughly an
e-puck on a large table.

Steps and seconds do different jobs and both stay. A run is replayed by step
count, archives record steps, and parity compares step for step; every physical
quantity is expressed per second, so `deltaTime` can change without changing
what a trial means. `testFixedStepIndependence` holds that invariant by driving
an agent into a wall for two simulated seconds at rates from 30 to 480 Hz.

## Adding a scenario

1. `src/worlds/scenarios/<Name>Scenario.cpp` fills in a `ScenarioDefinition`:
   brain shape, beacons, fitness, objective count, the optional before/after step
   hooks, the tunables the UI should offer, and the GPU parameter packer.
2. `shaders/worlds/<name>.glsl` unpacks the same parameter block for geometry,
   and `shaders/worlds/steps/<name>.glsl` mirrors the step hooks.
3. Add the enum value and one line to the registry in `src/worlds/WorldScenario.cpp`.

Shared math belongs in `include/vkexp/worlds/ScenarioKernel.inl`, which both
languages compile. Nothing else needs editing: the simulation, the scoring, the
batch runner's `--scenario` list and the UI controls all read the definition.

## Immediate next step

Add champion replay and receptor visualization before walls. Those tools make
later failures attributable to perception, control, fitness, or evolution
instead of only showing that population fitness stopped improving. The batch
runner and genome archives are in place, so a champion can now be evolved
headlessly and replayed in the window.
