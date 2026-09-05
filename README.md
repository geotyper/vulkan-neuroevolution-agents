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
vulkan_neuroevolution_agents (composition root)
  |
  +-- vkneuro_domain          no Vulkan dependency
  |     NeuralNetwork         flattened brain contract + CPU evaluator
  |     worlds/
  |       WorldScenario       public scenario dispatcher
  |       scenarios/          one C++ module per world scenario
  |     Sensors               CPU reference perception
  |     CpuSimulation         CPU reference physics/fitness
  |     GeneticAlgorithm      selection/crossover/mutation
  |
  +-- vkneuro_simulation
  |     SimulationModule      ping-pong state + compute orchestration
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
- `Topology` defines one flattened 1228-float weight layout used identically by
  the CPU evaluator and GLSL shader;
- module order in `main.cpp` is the composition graph: compute publishes the
  buffer, rendering reads it, and ImGui composites the viewport.

This lets a new sensor model, brain evaluator, selection policy, renderer, or
UI replace its counterpart without changing the application lifecycle.

## CPU/GPU correctness

`vkexp_compute_smoke` creates the same agent and genome on CPU and GPU, advances
both by one complete sensor/network/physics step, reads the SSBO back, and
compares every float with a small tolerance for both world shapes. A two-agent
GPU test verifies physical separation, tactile contact, and reception of an
emitted red signal, then verifies that the same colocated agents cannot collide
or exchange light across a logical-world boundary. Another parity case covers
the exact step at which the active beacon diagonal changes. Pure CPU tests cover
world scaling, beacon layouts, logical-world partition mapping, channel
mapping, weight layout, neural evaluation, elite preservation, and reusable
compute validation.

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

The debug suite contains pure unit tests, a CLI smoke test, and a headless
Vulkan test. The latter returns CTest's skip code when no compute device exists.

## Extension points

The next world feature should enter through a focused contract:

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
