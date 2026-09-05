# Vulkan Neuroevolution Agents

An experimental C++20/Vulkan playground for evolving thousands of small agent
brains on the GPU. The first scenario is deliberately simple: agents use seven
directional photoreceptors to find stationary or periodically changing beacon
positions, while a genetic algorithm evolves the weights of a fixed dense
neural network.

The repository is a working vertical slice and a base for later worlds with
walls, memory, inter-agent light perception, colonies, and richer fitness
functions. Simulation, evolution, visualization, UI, and Vulkan infrastructure
are separate targets rather than one application-specific module.

## Current experiment

- 512 genomes, each evaluated in four trials (2048 GPU agents);
- population partitioned into configurable logical groups (12 agents per world
  by default, with an all-agents mode);
- 7 forward light receptors with RGB and luminance channels;
- 8 full-body tactile sectors distinguishing walls from agents;
- `52 inputs -> 20 tanh neurons -> 8 outputs`;
- outputs control left/right motors, RGB emission, emission intensity, and two
  recurrent memory cells;
- inertial movement with linear/angular drag and hard linear/angular speed
  limits;
- selectable circular or square world in small (x1), medium (x1.5), and large
  (x3) sizes;
- stationary per-trial beacons, alternating diagonal pairs, orbiting beacons,
  deterministic random movement with occasional teleportation, or an
  orbiting-resource/static-home foraging cycle;
- circle-circle agent collisions with impulse response and tactile pressure;
- configurable accumulated fitness penalty for hitting or pushing against the
  world boundary;
- additive, softly tone-mapped RGB perception of nearby agent signals;
- runtime ablation switches for agent collisions and agent-light perception;
- average fitness across trials rewards progress and completion in every beacon
  phase and penalizes motor and signal energy use;
- elitism, tournament selection, uniform crossover, Gaussian mutation;
- adjustable simulation speed, generation length, agents per world, physics,
  and sensor FOV;
- generation length selectable from 120 to 15000 simulation steps;
- separate simulation, genetic-algorithm, world, and profiler windows;
- fitness/arrival history graphs for completed generations;
- an off-screen Vulkan renderer with persistent circular bodies, heading
  markers, and independently rendered coloured halos.

The population is divided into configurable groups, and each group keeps four
independent trial worlds. With the default group size of 12, 512 genomes occupy
43 groups and 172 logical worlds. A per-world uniform grid limits collision and
light queries to nearby agents in the same world. The viewport renders only the
selected world and can switch between groups and trials. Selecting all 512
agents produces one group and restores the original interaction density while
still avoiding the visual overlap of its four trials.

Individual fitness does not yet reward helping unrelated genomes, so the
communication channel is functional but meaningful signalling is not expected
until colony fitness is introduced.

The four trials are not four independently trained populations. Every genome
controls four agents with the same weights but different initial conditions.
Their scores are averaged before selection, which discourages solutions that
only work from one spawn position or heading.

## World and beacon scenarios

| Control | Variants | Behaviour |
| --- | --- | --- |
| World shape | Circle, Square | Changes the arena boundary and wall-contact response. |
| World size | Small x1, Medium x1.5, Large x3 | Scales the arena, spawn distribution, and beacon placement. Agent size, speed, and sensor range remain physical constants, so larger worlds are harder. |
| Arrival radius multiplier | x0.1 to x5.0 | Scales the fixed 0.060 beacon radius used for pickup/completion. At x1 the agent centre must enter the visible beacon circle. |
| Beacon scenario | Stationary | One fixed coloured target per logical trial. |
| Beacon scenario | Alternating diagonals | Two beacons occupy one diagonal during the first half of a generation, then two beacons occupy the opposite diagonal. |
| Beacon scenario | Rotating | One beacon per trial continuously orbits the world centre; speed and direction are adjustable from -90 to +90 degrees per second. |
| Beacon scenario | Random movement | Each trial follows a smooth bounded wandering path with adjustable speed and a configurable teleport chance checked every three seconds. |
| Beacon scenario | Forage + home | Agents collect an orange orbiting resource, then carry its decaying value to a blue home that relocates every eight seconds before seeking the resource again. |

Changing the world size, shape, or beacon scenario resets the evolution because
fitness values gathered in different environments are not directly comparable.
In the alternating scenario, reaching either active beacon completes the
current phase. Progress and completion are scored independently for both halves
of the generation. The moving scenarios additionally reward sustained visible
closeness, so following the target scores better than a chance encounter.
Rotating and random scenarios expose an orbit/roaming-radius control; the
default random teleport probability is 25%.

The foraging scenario does not grant passive tracking fitness. Reaching the
resource switches an explicit task input from `seek resource` to `seek home`
and fills a cargo-level input. Cargo decays while being carried, so prompt home
delivery is worth more; delivery completes a cycle and switches the task back.
The home teleports to a deterministic random position every eight simulation
seconds, independently for each trial and generation.
Fitness remains cumulative—the expiring cargo is the decreasing reward
potential—so long generations do not erase already completed work. Two separate
learned memory values are fed back as inputs on the next simulation step and
updated by the final two network outputs.

## Architecture

```text
vulkan_neuroevolution_agents (windowed composition root)
vkneuro_headless             (batch composition root)
  |
  +-- vkneuro_domain          no Vulkan dependency
  |     NeuralNetwork         flattened brain contract + CPU evaluator
  |     worlds/
  |       WorldScenario       scenario definition + public dispatcher
  |       scenarios/          world rules, brain shape, and fitness per scenario
  |     Sensors               CPU reference perception
  |     CpuSimulation         CPU reference physics/fitness
  |     GeneticAlgorithm      selection/crossover/mutation
  |
  +-- vkneuro_simulation
  |     SimulationDriver      population, GA, and per-step dispatch recording
  |     SimulationModule      frame-loop adapter over the driver
  |     worlds/*.glsl         shared per-scenario shader implementations
  |     agent_grid_*.comp     per-logical-world spatial acceleration
  |     agent_step.comp       RGB sensors + brain + collisions + physics
  |
  +-- vkneuro_visualization
  |     AgentRenderer         read-only consumer of the agent SSBO
  |
  +-- vkneuro_ui
  |     SimulationUiModule    controls/statistics/viewport only
  |
  +-- reusable infrastructure
        vkexp_core, vkexp_compute, vkexp_profiling, vkexp_imgui
```

The shared contracts are small:

- `SimulationState` carries controls, statistics, the published agent-buffer
  view, and the published viewport image;
- `AgentState` is an explicitly checked 176-byte std430-compatible structure;
- `Topology` defines the maximum 1228-float genome capacity, while every
  `ScenarioDefinition` selects its active input/hidden/output counts and fitness
  function; the CPU evaluator and GLSL shader use the same packed layout;
- module order in `main.cpp` is the composition graph: compute publishes the
  buffer, rendering reads it, and ImGui composites the viewport;
- `GpuStepParameters` carries only what every scenario needs, plus a 48-byte
  `ScenarioParameterBlock` each scenario packs and unpacks itself. It travels in
  a storage buffer indexed by step rather than in push constants, so adding a
  scenario neither widens a shared struct nor approaches the 128-byte push
  constant size Vulkan guarantees.

This lets a new sensor model, brain evaluator, selection policy, renderer, or
UI replace its counterpart without changing the application lifecycle.

`SimulationDriver` holds the experiment; `SimulationModule` only maps frame
callbacks onto it. The batch runner drives the same driver from an
`ImmediateContext`, so a sweep and the window run identical code.

The four beacon-following scenarios currently use a reactive `48 -> 20 -> 6`
brain. `Forage + home` declares `52 -> 20 -> 8`, adding task state and two
recurrent memory cells. Scenario changes still reset evolution, while the GA
and fixed-capacity GPU genome buffers remain shared.

## CPU/GPU correctness

`vkexp_compute_smoke` creates the same agent and genome on CPU and GPU, advances
both by one complete sensor/network/physics step, reads the SSBO back, and
compares every float with a small tolerance for both world shapes. A two-agent
GPU test verifies physical separation, tactile contact, and reception of an
emitted red signal, then verifies that the same colocated agents cannot collide
or exchange light across a logical-world boundary. Another parity case covers
the exact step at which the active beacon diagonal changes.

On top of that, every scenario runs a 540-step trajectory regression:

- **lockstep parity.** Each step feeds the CPU reference state to the GPU and
  compares one step of both, so the shader is checked at hundreds of genuinely
  reachable states -- including the alternating phase flip and the forage home
  relocation epoch -- without the chaotic drift a free-running trajectory would
  accumulate through tanh feedback.
- **accumulated drift budget.** Lockstep cannot see a systematic bias smaller
  than the per-step tolerance, because resetting to the CPU state each step stops
  it accumulating. Summing the signed per-step differences restores that:
  rounding noise cancels to ~1e-4 over 540 steps, while a changed shader constant
  reaches ~3e-2. The budget sits an order of magnitude above the noise.
- **coverage assertions.** A run whose beacon stayed out of sensor range, or whose
  alternating phase never flipped, fails rather than passing vacuously.
- **determinism.** 192 agents sharing one spatial grid are stepped twice; the
  results must be bit-identical, which is where a grid race would surface.

Pure CPU tests cover world scaling, beacon layouts, logical-world partition
mapping, channel mapping, weight layout, neural evaluation, elite preservation,
scenario parameter packing, resolved step settings, genome archive round-trips
including corruption and truncation rejection, and reusable compute validation.

## Build and run

Requirements: CMake 3.24+, Ninja, a C++20 compiler, Vulkan 1.3 development
files and driver, GLFW 3.3+, `glslangValidator`, and X11/Wayland for the GUI.
Dear ImGui v1.91.8 is fetched by CMake.

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
./build/debug/vulkan_neuroevolution_agents
```

Disable validation if the validation layer is unavailable:

```bash
./build/debug/vulkan_neuroevolution_agents --no-validation
```

### Batch runs

`vkneuro_headless` evolves without a window, which is what makes overnight runs,
parameter sweeps and ablation comparisons possible:

```bash
./build/release/vkneuro_headless --scenario rotating --generations 200 \
    --csv runs/rotating.csv --save-champion runs/rotating-champion.vkng

# Signal-off ablation against the same seed.
./build/release/vkneuro_headless --scenario forage --generations 200 --seed 7 \
    --no-agent-light --csv runs/forage-nolight.csv

# Resume a saved population.
./build/release/vkneuro_headless --scenario forage --generations 50 \
    --load-population runs/forage-population.vkng
```

`--help` lists every option. Genome archives are versioned little-endian files
that record the generation, scenario, seed, fitness and brain shape, and refuse
to load into a build with a different weight count.

The debug suite contains pure unit tests, CLI smoke tests, two short real
headless evolution runs covering the archive round trip, and a headless Vulkan
parity test. The Vulkan-dependent ones return CTest's skip code when no compute
device exists.

## Extension points

The next world feature should enter through a focused contract:

- a new scenario adds one `.cpp`/`.glsl` pair plus its parameter block; it does
  not touch the shared step parameters;
- walls and occlusion extend sensor/world queries;
- richer recurrent cells or gated memory can extend the two-value recurrent state;
- internal walls and occlusion extend grid-backed world queries;
- colony scoring replaces fitness aggregation without changing physics;
- a different topology can become another evaluator/shader pair;
- GPU-side evolution can later replace the synchronous generation boundary.

## Project documentation

- [PROGRESS.md](PROGRESS.md) records what was implemented at each completed
  stage, the decisions behind it, and the verification status.
- [PLAN.md](PLAN.md) is the forward-looking roadmap and list of unfinished
  milestones.
